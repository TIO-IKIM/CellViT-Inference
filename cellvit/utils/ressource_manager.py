# -*- coding: utf-8 -*-
import os
import psutil
import subprocess
import re
import logging
import ray
from typing import Dict, Any, Optional, List, Tuple, Literal
from cellvit.utils.logger import NullLogger
from cellvit.utils.check_module import check_module
from cellvit.utils.check_cupy import check_cupy


def detect_runtime_environment() -> str:
    """Detect the runtime environment (SLURM, Kubernetes, Docker, VM, or server).

    Returns:
        str: The detected runtime environment.
    """
    if is_slurm():
        return "slurm"
    if is_docker():
        return "docker"
    if is_kubernetes():
        return "kubernetes"
    if is_vm():
        return "vm"
    return "server"


def is_slurm() -> bool:
    """Check if running inside a SLURM job

    Returns:
        bool: True if running inside a SLURM job, False otherwise.
    """
    return any(
        var in os.environ
        for var in ["SLURM_JOB_ID", "SLURM_NODEID", "SLURM_CLUSTER_NAME"]
    )


def is_kubernetes() -> bool:
    """Check if running inside a Kubernetes pod.

    Returns:
        bool: True if running inside a Kubernetes pod, False otherwise.
    """
    return (
        os.path.exists("/var/run/secrets/kubernetes.io/")
        or "KUBERNETES_SERVICE_HOST" in os.environ
    )


def is_docker() -> bool:
    """Check if running inside a Docker container.

    Returns:
        bool: True if running inside a Docker container, False otherwise.
    """
    if is_kubernetes():
        return False  # Prioritize Kubernetes detection

    cgroup_paths = ["/proc/self/cgroup", "/proc/1/cgroup"]
    for path in cgroup_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    if any("docker" in line or "containerd" in line for line in f):
                        return True
            except:
                pass
    return False


def is_vm() -> bool:
    """Check if running inside a virtual machine or hypervisor.

    Returns:
        bool: True if running inside a VM, False otherwise.
    """
    if is_docker() or is_kubernetes():
        return False  # Containers take priority

    try:
        output = subprocess.check_output("systemd-detect-virt", text=True).strip()
        if output and output not in ("none", ""):
            return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return False


def get_cpu_memory_slurm() -> Tuple[float, float]:
    """Get CPU and memory limits from a SLURM job.

    Returns:
        Tuple[float, float]: CPU count and memory limit in MB.
    """
    cpu_count, memory_mb = None, None

    # List of SLURM environment variables to check for CPU count
    cpu_env_vars = [
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_NTASKS",
        "SLURM_TASKS_PER_NODE",
    ]

    # List of SLURM environment variables to check for memory
    mem_env_vars = ["SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU"]

    # Try to fetch from SLURM command if SLURM_JOB_ID is available
    if "SLURM_JOB_ID" in os.environ:
        try:
            slurm_output = subprocess.check_output(
                ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]],
                text=True,
                stderr=subprocess.DEVNULL,
            )

            # Parse CPU information
            cpus_match = re.search(r"NumCPUs=(\d+)", slurm_output)
            cpus_per_task_match = re.search(r"CPUs/Task=(\d+)", slurm_output)

            if cpus_match:
                cpu_count = int(cpus_match.group(1))
            elif cpus_per_task_match:
                cpu_count = int(cpus_per_task_match.group(1))

            # Parse memory information (with multiple possible formats)
            mem_match = re.search(r"MinMemory=(\d+)([KMGT])?", slurm_output)
            mem_per_cpu_match = re.search(r"MinMemoryCPU=(\d+)([KMGT])?", slurm_output)
            mem_per_node_match = re.search(
                r"MinMemoryNode=(\d+)([KMGT])?", slurm_output
            )

            # Try different memory specifications in order
            for match in [mem_match, mem_per_cpu_match, mem_per_node_match]:
                if match:
                    memory_value = int(match.group(1))
                    unit = match.group(2) if len(match.groups()) > 1 else None

                    # Convert to MB based on unit
                    if unit == "K":
                        memory_mb = memory_value / 1024
                    elif unit == "G":
                        memory_mb = memory_value * 1024
                    elif unit == "T":
                        memory_mb = memory_value * 1024 * 1024
                    else:  # Default is MB or no unit specified
                        memory_mb = memory_value

                    # If it's memory per CPU, multiply by CPU count
                    if match == mem_per_cpu_match and cpu_count:
                        memory_mb *= cpu_count

                    break  # Stop after first successful match

        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
            # Silently fall back to environment variables if scontrol fails
            pass

    # Fallback to SLURM environment variables for CPU count
    if cpu_count is None:
        for var in cpu_env_vars:
            if var in os.environ:
                try:
                    value = os.environ[var]
                    # Handle formats like "4(x2)" which means 4 CPUs on 2 nodes
                    if "(" in value:
                        value = value.split("(")[0]
                    cpu_count = int(value)
                    break
                except (ValueError, TypeError):
                    continue

    # Fallback to SLURM environment variables for memory
    if memory_mb is None:
        for var in mem_env_vars:
            if var in os.environ:
                try:
                    value = os.environ[var]
                    memory_mb = int(value)

                    # If it's memory per CPU, multiply by CPU count if available
                    if var == "SLURM_MEM_PER_CPU" and cpu_count:
                        memory_mb *= cpu_count
                    break
                except (ValueError, TypeError):
                    continue

    # Final fallback to system resources if all else fails
    if cpu_count is None:
        # Try to use job-restricted CPU affinity first
        try:
            if hasattr(os, "sched_getaffinity"):
                cpu_count = len(os.sched_getaffinity(0))
            else:
                cpu_count = psutil.cpu_count(logical=True)
        except:
            cpu_count = psutil.cpu_count(logical=True)

    if memory_mb is None:
        try:
            # Use SLURM-restricted memory if possible
            memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        except:
            # Last resort fallback
            memory_mb = 1024  # Assume 1GB as absolute minimum

    return cpu_count, memory_mb


def get_cpu_memory_kubernetes() -> Tuple[float, float]:
    """Get CPU and memory limits from a kubernetes pod.

    Returns:
        Tuple[float, float]: CPU count and memory limit in MB.
    """
    cpu_count, memory_mb = None, None

    # First try Downward API environment variables (if configured in pod spec)
    k8s_cpu_limit = os.environ.get("MY_CPU_LIMIT") or os.environ.get("CPU_LIMIT")
    k8s_mem_limit = os.environ.get("MY_MEM_LIMIT") or os.environ.get("MEMORY_LIMIT")

    if k8s_cpu_limit:
        # Kubernetes CPU format is "100m" for 0.1 CPU, or "1" for 1 CPU
        if "m" in k8s_cpu_limit:
            cpu_count = (
                float(k8s_cpu_limit.replace("m", "")) / 1000
            )  # Convert "100m" to 0.1 CPU
        else:
            cpu_count = float(k8s_cpu_limit)

    if k8s_mem_limit:
        # Kubernetes memory format: "1Gi", "512Mi", "1024Ki", etc.
        if k8s_mem_limit.endswith("Ki"):
            memory_mb = float(k8s_mem_limit[:-2]) / 1024  # Convert Ki to Mi
        elif k8s_mem_limit.endswith("Mi"):
            memory_mb = float(k8s_mem_limit[:-2])
        elif k8s_mem_limit.endswith("Gi"):
            memory_mb = float(k8s_mem_limit[:-2]) * 1024  # Convert Gi to Mi
        else:
            try:
                memory_mb = float(k8s_mem_limit) / (
                    1024 * 1024
                )  # Try to convert if it's a large number
            except ValueError:
                pass

    # If environment variables don't work, try to read from cgroups (similar to Docker)
    if cpu_count is None or memory_mb is None:
        # Check cgroup v1 & v2 CPU limits
        cpu_quota, cpu_period = None, None
        cpu_quota_paths = [
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
            "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
            "/sys/fs/cgroup/cpu.max",  # cgroup v2
        ]
        cpu_period_paths = [
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
            "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
        ]

        for path in cpu_quota_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        content = f.read().strip()
                        if path.endswith("cpu.max"):
                            parts = content.split()
                            if parts[0] != "max":
                                cpu_quota, cpu_period = int(parts[0]), int(parts[1])
                        else:
                            cpu_quota = int(content)
                except:
                    pass

        if cpu_period is None and cpu_quota is not None:
            for path in cpu_period_paths:
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            cpu_period = int(f.read().strip())
                    except:
                        pass

        if cpu_quota and cpu_period and cpu_quota > 0:
            cpu_count = cpu_count or (cpu_quota / cpu_period)

        # Check cgroup memory limits
        memory_paths = [
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory.max",  # cgroup v2
        ]
        for path in memory_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        memory_limit = int(f.read().strip())
                        if (
                            memory_limit < 9223372036854000000
                        ):  # Ignore if it's effectively unlimited
                            memory_mb = memory_mb or (memory_limit / (1024 * 1024))
                except:
                    pass

    # Last resort: use container limits via cgroups
    if cpu_count is None:
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else psutil.cpu_count(logical=True)
        )

    if memory_mb is None:
        memory_mb = psutil.virtual_memory().total / (1024 * 1024)

    return cpu_count, memory_mb


def get_cpu_memory_docker() -> Tuple[float, float]:
    """Get CPU and memory limits from a docker container job.

    Returns:
        Tuple[float, float]: CPU count and memory limit in MB.
    """
    cpu_quota, cpu_period, memory_limit = None, None, None

    # Check cgroup v1 & v2 CPU limits
    cpu_quota_paths = [
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu.max",  # cgroup v2
    ]
    for path in cpu_quota_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    content = f.read().strip()
                    if path.endswith("cpu.max"):
                        parts = content.split()
                        if parts[0] != "max":
                            cpu_quota, cpu_period = int(parts[0]), int(parts[1])
                    else:
                        cpu_quota = int(content)
            except:
                pass

    if cpu_quota and cpu_period and cpu_quota > 0:
        cpu_count = cpu_quota / cpu_period
    else:
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else psutil.cpu_count()
        )

    # Check cgroup memory limits
    memory_paths = [
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory.max",  # cgroup v2
    ]
    for path in memory_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    memory_limit = int(f.read().strip())
                    if (
                        memory_limit > 9223372036854000000
                    ):  # Ignore if it's effectively unlimited
                        memory_limit = None
            except:
                pass

    memory_mb = (
        memory_limit / (1024 * 1024)
        if memory_limit
        else psutil.virtual_memory().total / (1024 * 1024)
    )

    return cpu_count, memory_mb


def get_cpu_memory_vm_or_server() -> Tuple[float, float]:
    """Get CPU and memory limits from a VM or a bare-metal server

    Returns:
        Tuple[float, float]: CPU count and memory limit in MB.
    """
    return psutil.cpu_count(logical=True), psutil.virtual_memory().total / (1024 * 1024)


def get_cpu_resources(logger: Optional[logging.Logger] = None):
    """Returns the number of available CPU cores and memory (MB) for the current runtime environment."""
    logger = logger or NullLogger()

    env = detect_runtime_environment()

    logger.info(f"Environment: {env}")

    if env == "slurm":
        cpu_stats = get_cpu_memory_slurm()
    elif env == "kubernetes":
        cpu_stats = get_cpu_memory_kubernetes()
    elif env == "docker":
        cpu_stats = get_cpu_memory_docker()
    else:
        cpu_stats = get_cpu_memory_vm_or_server()

    logger.info(f"Available cores: {cpu_stats[0]}")
    logger.info(f"Available memory: {cpu_stats[1]/1024} (GB)")
    return cpu_stats, env  # Covers "vm" and "server"


def get_gpu_resources(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Get GPU resources available to the current process.
    Uses PyTorch to check CUDA availability and properties.

    Args:
        logger: Optional logger to use for logging messages

    Returns:
        dict: A dictionary containing GPU information or None if no GPU is available
        gpu_resources = {
            'has_gpu': False,
            'gpu_count': 0,
            'details': {},
            'devices': {}
        }
    """
    logger = logger or NullLogger()
    # Use a default logger if none is provided
    logger.info("Checking GPU availability...")

    gpu_resources = {"has_gpu": False, "gpu_count": 0, "details": {}, "devices": {}}

    try:
        import torch

        # Check if CUDA is available
        use_cuda = torch.cuda.is_available()
        gpu_resources["has_gpu"] = use_cuda

        if not use_cuda:
            logger.warning("No CUDA-capable GPU detected.")
            return gpu_resources

        # Get basic GPU information
        gpu_resources["gpu_count"] = torch.cuda.device_count()
        gpu_resources["details"]["cudnn_version"] = (
            torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
        )

        for i in range(gpu_resources["gpu_count"]):
            device_info = {
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": torch.cuda.get_device_properties(i).total_memory
                / 1e9,
            }

            # Try to get additional information from device properties
            props = torch.cuda.get_device_properties(i)
            device_info["compute_capability"] = f"{props.major}.{props.minor}"

            # Try to get memory usage (this may not work in all environments)
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                memory_reserved = torch.cuda.memory_reserved(i) / 1e9
                device_info["memory_allocated_gb"] = memory_allocated
                device_info["memory_reserved_gb"] = memory_reserved
                device_info["memory_free_gb"] = (
                    device_info["total_memory_gb"] - memory_allocated
                )
            except:
                # Memory usage information not available
                pass

            gpu_resources["devices"][i] = device_info

        # Log success and device information
        logger.info("CUDA-capable GPU detected.")
        logger.info(
            f"CUDNN Version: {gpu_resources['details'].get('cudnn_version', 'Not available')}"
        )
        logger.info(f"Number of CUDA Devices: {gpu_resources['gpu_count']}")

        # Log information about the first GPU
        if gpu_resources["gpu_count"] > 0:
            device0 = gpu_resources["devices"][0]
            logger.info(f"CUDA Device 0 Name: {device0['name']}")
            logger.info(
                f"CUDA Device 0 Total Memory: {device0['total_memory_gb']:.2f} GB"
            )

    except ImportError:
        logger.error("PyTorch not installed. Cannot check GPU availability.")
        gpu_resources["details"]["error"] = "PyTorch not installed"
    except Exception as e:
        logger.error(f"Unexpected error during GPU check: {str(e)}")
        gpu_resources["details"]["error"] = str(e)

    return gpu_resources


def get_used_memory(runtime_env: str) -> float:
    """Get the current memory usage in MB for the given runtime environment.

    Args:
        runtime_env (str): The runtime environment (slurm, kubernetes, docker, vm, server).

    Returns:
        float: The current memory usage in MB.
    """
    if runtime_env == "slurm":
        return get_used_memory_slurm()
    elif runtime_env == "kubernetes":
        return get_used_memory_kubernetes()
    elif runtime_env == "docker":
        return get_used_memory_docker()
    else:  # vm or server
        return get_used_memory_process()


def get_used_memory_process() -> float:
    """Get memory usage for the current process and its children.

    Returns:
        float: The total memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    # Get memory for this process and all its children recursively
    memory_info = process.memory_info()

    # Also sum children processes memory
    children_mem = 0
    try:
        for child in process.children(recursive=True):
            try:
                children_mem += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    # Return total memory in MB
    return (memory_info.rss + children_mem) / (1024 * 1024)


def get_used_memory_kubernetes() -> float:
    """Get memory usage within a Kubernetes pod.

    Returns:
        float: The current memory usage in MB.
    """
    # Try to read from cgroup memory.usage_in_bytes first
    memory_paths = [
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory.current",  # cgroup v2
    ]

    for path in memory_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    # Convert from bytes to MB
                    return int(f.read().strip()) / (1024 * 1024)
            except:
                pass

    # Fallback to process-based calculation
    return get_used_memory_process()


def get_used_memory_docker() -> float:
    """Get memory usage within a Docker container.

    Returns:
        float: The current memory usage in MB.
    """
    # Same approach as Kubernetes since both use cgroups
    return get_used_memory_kubernetes()


def get_used_memory_slurm() -> float:
    """Get memory usage within a SLURM job.

    Returns:
        float: The current memory usage in MB.
    """
    if "SLURM_JOB_ID" in os.environ:
        try:
            # Try using sstat command to get memory usage
            job_id = os.environ["SLURM_JOB_ID"]
            slurm_output = subprocess.check_output(
                ["sstat", "--format=MaxRSS", "-j", job_id, "--noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            # sstat returns memory in KB format like "1024K"
            if slurm_output:
                if "K" in slurm_output:
                    return (
                        float(slurm_output.replace("K", "")) / 1024
                    )  # Convert KB to MB
                else:
                    try:
                        return float(slurm_output) / 1024  # Assume KB if no unit
                    except ValueError:
                        pass
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    # If SLURM-specific methods fail, try the cgroup approach (SLURM may use cgroups)
    memory_paths = [
        "/sys/fs/cgroup/memory/slurm/uid_{}/job_{}/memory.usage_in_bytes".format(
            os.getuid(), os.environ.get("SLURM_JOB_ID", "")
        ),
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory.current",  # cgroup v2
    ]

    for path in memory_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return int(f.read().strip()) / (1024 * 1024)  # Convert to MB
            except:
                pass

    # If all SLURM and cgroup methods fail, fall back to process-based
    return get_used_memory_process()


def retrieve_actor_usage() -> List[float]:
    actor_pids = [f["Pid"] for f in ray._private.state.actors().values()]
    memory_usage = []
    for actor_pid in actor_pids:
        try:
            mem = psutil.Process(actor_pid).memory_info().rss / (1024 * 1024)
            memory_usage.append(mem)
        except:
            pass

    return memory_usage


class SystemConfiguration:
    def __init__(self, gpu: int = 0) -> None:
        """Initialize the SystemConfiguration object.

        Args:
            gpu (int, optional): CUDA ID for the GPU. Defaults to 0.

        Raises:
            SystemError: Requesting non existing gpu index

        Attributes:
            runtime_environment (str): The runtime environment (slurm, kubernetes, docker, vm, server).
            cpu_count (int): The number of available CPU cores.
            memory (float): The total memory available in GB.
            has_gpu (bool): True if a GPU is available, False otherwise.
            gpu_count (int): The number of available GPUs.
            gpu_memory (float): The total memory available on the selected GPU in GB.
            ray (bool): True if Ray is available, False otherwise.
            cupy (bool): True if CuPy is available, False otherwise.
            cucim (bool): True if CuCIM is available, False otherwise.
            numba (bool): True if Numba is available, False otherwise.
            ray_worker (int): The number of Ray workers that can be created.
            ray_remote_cpus (int): The number of CPUs per Ray worker.
            torch_worker (int): The number of Torch workers that can be created.
            gpu_index (int): The index of the selected GPU.

        Methods:
            __getitem__(key: str) -> Any: Get an attribute by key.
            _calculate_ray_worker() -> None: Calculate the number of Ray workers.
            overwrite_ray_worker(worker_count: int) -> None: Overwrite the number of Ray workers.
            overwrite_ray_remote_cpus(ray_remote_cpus: int) -> None: Overwrite the number of CPUs per Ray worker.
            overwrite_available_cpus(cpu_count: int) -> None: Overwrite the number of available CPUs.
            overwrite_memory(memory: int) -> None: Overwrite the total memory available.
            get_current_memory_usage() -> int: Get the current memory usage.
            get_current_memory_percentage() -> int: Get the current memory usage percentage.
            log_system_configuration(logger: Optional[logging.Logger] = None) -> None: Log the system configuration
        """
        self.runtime_environment: Literal["slurm, kubernetes, docker, vm, server"]
        self.cpu_count: int
        self.memory: float
        self.has_gpu: bool
        self.gpu_count: int
        self.gpu_memory: float
        self.ray: bool
        self.cupy: bool
        self.cucim: bool
        self.numba: bool
        self.ray_worker: int
        self.ray_remote_cpus: int
        self.torch_worker: int
        self.gpu_index: int = gpu

        (cpu_count, memory), env = get_cpu_resources()
        gpu_resources = get_gpu_resources()

        self.cpu_count = int(cpu_count)
        self.memory = memory
        self.runtime_environment = env
        self.has_gpu = gpu_resources["has_gpu"]
        self.gpu_count = gpu_resources["gpu_count"]
        if self.gpu_index >= self.gpu_count:
            raise SystemError("Requesting non existing gpu index")
        self.gpu_memory = gpu_resources["devices"][self.gpu_index]["total_memory_gb"]

        self.cupy = check_module("cupy") or check_module("cupyx")
        if self.cupy:
            self.cupy = check_cupy(True, NullLogger())
        self.cucim = check_module("cucim")
        self.numba = check_module("numba")
        self.ray = check_module("ray")
        self._calculate_ray_worker()

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in SystemConfiguration")

    def _calculate_ray_worker(self) -> None:
        # TODO: Make some tests and adapt it
        # each ray worker needs 4-8 cpus
        available_ray_cpus = self.cpu_count - 2
        if available_ray_cpus <= 12:
            self.ray_remote_cpus = 4
        if available_ray_cpus < 16:
            self.ray_remote_cpus = 6
        else:
            self.ray_remote_cpus = 8

        self.ray_worker = int(available_ray_cpus / self.ray_remote_cpus)

    def overwrite_ray_worker(self, worker_count: int) -> None:
        if self.ray_worker >= 10:
            ray_worker = 9
        self.ray_worker = worker_count
        self.ray_remote_cpus = int((self.cpu_count - 2) / self.ray_worker)

    def overwrite_ray_remote_cpus(self, ray_remote_cpus: int) -> None:
        self.ray_remote_cpus = ray_remote_cpus
        available_ray_cpus = self.cpu_count - 2
        self.ray_worker = int(available_ray_cpus / self.ray_remote_cpus)

    def overwrite_available_cpus(self, cpu_count: int) -> None:
        self.cpu_count = int(cpu_count)
        self._calculate_ray_worker()

    def overwrite_memory(self, memory: int) -> None:
        self.memory = memory

    def get_current_memory_usage(self) -> int:
        return int(get_used_memory(self.runtime_environment))

    def get_current_memory_percentage(self) -> int:
        return (get_used_memory(self.runtime_environment) / self.memory) * 100

    def log_system_configuration(self, logger: logging.Logger = None) -> None:
        if logger is None:
            logger = logging.getLogger("SystemConfiguration")
            logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(levelname)-8s | %(name)-20s | %(message)s")
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        logger.info("========================================")
        logger.info("         SYSTEM CONFIGURATION           ")
        logger.info("========================================")

        logger.info("System Configuration:")
        logger.info(f"CPU count:          {self.cpu_count}")
        logger.info(f"Memory:             {self.memory / 1024:.2f} GB")
        logger.info(f"GPU count:          {self.gpu_count}")
        logger.info(f"Used GPU-ID:        {self.gpu_index}")
        logger.info(f"GPU memory:         {self.gpu_memory:.2f} GB")
        logger.info(f"Ray available:      {self.ray}")
        logger.info(f"Ray worker count:   {self.ray_worker}")
        logger.info(f"Ray remote cpus:    {self.ray_remote_cpus}")
        logger.info(f"Cupy available:     {self.cupy}")
        logger.info(f"Cucim available:    {self.cucim}")
        logger.info(f"Numba available:    {self.numba}")

        logger.info("========================================")
        logger.info("       SYSTEM LOADED SUCCESSFULLY       ")
        logger.info("========================================")

        # Remove handlers and destroy logger after use
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        print("\n\n")
