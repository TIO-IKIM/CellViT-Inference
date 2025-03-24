# -*- coding: utf-8 -*-
# Test  Ressource Manager
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os
import unittest
from unittest.mock import patch, mock_open, MagicMock
from unittest import TestCase

from cellvit.utils.ressource_manager import (
    detect_runtime_environment,
    is_slurm,
    is_kubernetes,
    is_docker,
    is_vm,
    get_cpu_memory_slurm,
    get_cpu_memory_kubernetes,
    get_cpu_resources,
    get_cpu_memory_vm_or_server,
    get_gpu_resources,
    get_used_memory,
    get_used_memory_kubernetes,
    get_used_memory_process,
    get_used_memory_slurm,
    SystemConfiguration,
)


class TestRessourceManager(unittest.TestCase):
    @patch.dict(os.environ, {"SLURM_JOB_ID": "12345"})
    def test_is_slurm_true(self):
        """Test is_slurm when SLURM_JOB_ID is set."""
        self.assertTrue(is_slurm())

    @patch.dict(os.environ, {}, clear=True)
    def test_is_slurm_false(self):
        """Test is_slurm when SLURM_JOB_ID is not set."""
        self.assertFalse(is_slurm())

    @patch("os.path.exists", return_value=True)
    @patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "localhost"})
    def test_is_kubernetes_true(self, mock_exists):
        """Test is_kubernetes when /proc/self/cgroup exists and contains 'kubelet'."""
        self.assertTrue(is_kubernetes())

    @patch("os.path.exists", return_value=False)
    @patch.dict(os.environ, {}, clear=True)
    def test_is_kubernetes_false(self, mock_exists):
        """Test is_kubernetes when /proc/self/cgroup does not exist."""
        self.assertFalse(is_kubernetes())

    @patch("os.path.exists", side_effect=lambda path: path == "/proc/self/cgroup")
    @patch("builtins.open", new_callable=mock_open, read_data="docker")
    @patch("cellvit.utils.ressource_manager.is_kubernetes", return_value=False)
    def test_is_docker_true(self, mock_kubernetes, mock_open, mock_exists):
        """Test is_docker when /proc/self/cgroup exists and contains 'docker'."""
        self.assertTrue(is_docker())

    @patch("os.path.exists", return_value=False)
    @patch("cellvit.utils.ressource_manager.is_kubernetes", return_value=False)
    def test_is_docker_false(self, mock_kubernetes, mock_exists):
        """Test is_docker when /proc/self/cgroup does not exist."""
        self.assertFalse(is_docker())

    @patch("subprocess.check_output", return_value="kvm\n")
    @patch("cellvit.utils.ressource_manager.is_docker", return_value=False)
    @patch("cellvit.utils.ressource_manager.is_kubernetes", return_value=False)
    def test_is_vm_true(self, mock_kubernetes, mock_docker, mock_subprocess):
        """Test is_vm when subprocess.check_output returns 'kvm'."""
        self.assertTrue(is_vm())

    @patch("subprocess.check_output", side_effect=FileNotFoundError)
    @patch("cellvit.utils.ressource_manager.is_docker", return_value=False)
    @patch("cellvit.utils.ressource_manager.is_kubernetes", return_value=False)
    def test_is_vm_false(self, mock_kubernetes, mock_docker, mock_subprocess):
        """Test is_vm when subprocess.check_output raises FileNotFoundError."""
        self.assertFalse(is_vm())

    @patch("cellvit.utils.ressource_manager.is_slurm", return_value=True)
    def test_detect_runtime_environment_slurm(self, mock_slurm):
        """Test detect_runtime_environment for Slurm."""
        self.assertEqual(detect_runtime_environment(), "slurm")

    @patch("cellvit.utils.ressource_manager.is_docker", return_value=True)
    @patch("cellvit.utils.ressource_manager.is_slurm", return_value=False)
    def test_detect_runtime_environment_docker(self, mock_slurm, mock_docker):
        """Test detect_runtime_environment for Docker."""
        self.assertEqual(detect_runtime_environment(), "docker")

    @patch("cellvit.utils.ressource_manager.is_kubernetes", return_value=True)
    @patch("cellvit.utils.ressource_manager.is_slurm", return_value=False)
    def test_detect_runtime_environment_kubernetes(self, mock_slurm, mock_kubernetes):
        """Test detect_runtime_environment for Kubernetes."""
        self.assertEqual(detect_runtime_environment(), "kubernetes")

    @patch("cellvit.utils.ressource_manager.is_vm", return_value=True)
    @patch("cellvit.utils.ressource_manager.is_slurm", return_value=False)
    def test_detect_runtime_environment_vm(self, mock_slurm, mock_vm):
        """Test detect_runtime_environment for VM."""
        self.assertEqual(detect_runtime_environment(), "vm")

    @patch("cellvit.utils.ressource_manager.is_vm", return_value=False)
    @patch("cellvit.utils.ressource_manager.is_slurm", return_value=False)
    def test_detect_runtime_environment_server(self, mock_slurm, mock_vm):
        """Test detect_runtime_environment for server."""
        self.assertEqual(detect_runtime_environment(), "server")


class TestRessourceManagerCPU(TestCase):
    @patch.dict(os.environ, {"SLURM_JOB_ID": "12345"})
    @patch("subprocess.check_output", return_value="NumCPUs=4\nMinMemory=8G\n")
    def test_get_cpu_memory_slurm_with_scontrol(self, mock_subprocess):
        """Test get_cpu_memory_slurm with scontrol."""
        cpu_count, memory_mb = get_cpu_memory_slurm()
        self.assertEqual(cpu_count, 4)
        self.assertEqual(memory_mb, 8192)

    @patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": "4096"})
    def test_get_cpu_memory_slurm_with_env_vars(self):
        """Test get_cpu_memory_slurm with environment variables."""
        cpu_count, memory_mb = get_cpu_memory_slurm()
        self.assertEqual(cpu_count, 2)
        self.assertEqual(memory_mb, 4096)

    @patch.dict(os.environ, {}, clear=True)
    @patch("psutil.cpu_count", return_value=8)
    @patch("psutil.virtual_memory")
    def test_get_cpu_memory_slurm_fallback(self, mock_virtual_memory, mock_cpu_count):
        """Test get_cpu_memory_slurm with fallback to psutil."""
        mock_virtual_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16 GB
        cpu_count, memory_mb = get_cpu_memory_slurm()
        self.assertEqual(cpu_count, 8)
        self.assertEqual(memory_mb, 16384)

    @patch.dict(os.environ, {"MY_CPU_LIMIT": "200m", "MY_MEM_LIMIT": "512Mi"})
    def test_get_cpu_memory_kubernetes_with_env_vars(self):
        """Test get_cpu_memory_kubernetes with environment variables."""
        cpu_count, memory_mb = get_cpu_memory_kubernetes()
        self.assertAlmostEqual(cpu_count, 0.2)
        self.assertEqual(memory_mb, 512)

    @patch(
        "os.path.exists",
        side_effect=lambda path: path
        in [
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        ],
    )
    @patch(
        "builtins.open",
        side_effect=lambda path, *args, **kwargs: mock_open(
            read_data="100000\n"
        ).return_value
        if path == "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        else mock_open(read_data="100000\n").return_value
        if path == "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
        else mock_open().return_value,
    )
    @patch("psutil.cpu_count", return_value=4)
    def test_get_cpu_memory_kubernetes_with_cgroups(
        self, mock_cpu_count, mock_open, mock_exists
    ):
        """Test get_cpu_memory_kubernetes with cgroups."""
        cpu_count, memory_mb = get_cpu_memory_kubernetes()
        self.assertEqual(cpu_count, 1)  # 100000 quota / 100000 period = 1 CPU
        self.assertIsNotNone(memory_mb)  # Memory fallback to psutil

    @patch("psutil.cpu_count", return_value=16)
    @patch("psutil.virtual_memory")
    def test_get_cpu_memory_vm_or_server(self, mock_virtual_memory, mock_cpu_count):
        mock_virtual_memory.return_value.total = 32 * 1024 * 1024 * 1024  # 32 GB
        cpu_count, memory_mb = get_cpu_memory_vm_or_server()
        self.assertEqual(cpu_count, 16)
        self.assertEqual(memory_mb, 32768)


class TestGetCPUResourceFunction(TestCase):
    @patch(
        "cellvit.utils.ressource_manager.detect_runtime_environment",
        return_value="slurm",
    )
    @patch(
        "cellvit.utils.ressource_manager.get_cpu_memory_slurm", return_value=(4, 8192)
    )
    @patch("cellvit.utils.ressource_manager.NullLogger")
    def test_get_cpu_resources_slurm(
        self, mock_logger, mock_get_cpu_memory_slurm, mock_detect_env
    ):
        """Test get_cpu_resources for Slurm environment."""
        logger = mock_logger()
        cpu_stats, env = get_cpu_resources(logger)
        self.assertEqual(cpu_stats, (4, 8192))
        self.assertEqual(env, "slurm")
        logger.info.assert_any_call("Environment: slurm")
        logger.info.assert_any_call("Available cores: 4")
        logger.info.assert_any_call("Available memory: 8.0 (GB)")

    @patch(
        "cellvit.utils.ressource_manager.detect_runtime_environment",
        return_value="kubernetes",
    )
    @patch(
        "cellvit.utils.ressource_manager.get_cpu_memory_kubernetes",
        return_value=(2, 4096),
    )
    @patch("cellvit.utils.ressource_manager.NullLogger")
    def test_get_cpu_resources_kubernetes(
        self, mock_logger, mock_get_cpu_memory_kubernetes, mock_detect_env
    ):
        """Test get_cpu_resources for Kubernetes environment."""
        logger = mock_logger()
        cpu_stats, env = get_cpu_resources(logger)
        self.assertEqual(cpu_stats, (2, 4096))
        self.assertEqual(env, "kubernetes")
        logger.info.assert_any_call("Environment: kubernetes")
        logger.info.assert_any_call("Available cores: 2")
        logger.info.assert_any_call("Available memory: 4.0 (GB)")

    @patch(
        "cellvit.utils.ressource_manager.detect_runtime_environment",
        return_value="docker",
    )
    @patch(
        "cellvit.utils.ressource_manager.get_cpu_memory_docker", return_value=(8, 16384)
    )
    @patch("cellvit.utils.ressource_manager.NullLogger")
    def test_get_cpu_resources_docker(
        self, mock_logger, mock_get_cpu_memory_docker, mock_detect_env
    ):
        """Test get_cpu_resources for Docker environment."""
        logger = mock_logger()
        cpu_stats, env = get_cpu_resources(logger)
        self.assertEqual(cpu_stats, (8, 16384))
        self.assertEqual(env, "docker")
        logger.info.assert_any_call("Environment: docker")
        logger.info.assert_any_call("Available cores: 8")
        logger.info.assert_any_call("Available memory: 16.0 (GB)")

    @patch(
        "cellvit.utils.ressource_manager.detect_runtime_environment", return_value="vm"
    )
    @patch(
        "cellvit.utils.ressource_manager.get_cpu_memory_vm_or_server",
        return_value=(16, 32768),
    )
    @patch("cellvit.utils.ressource_manager.NullLogger")
    def test_get_cpu_resources_vm(
        self, mock_logger, mock_get_cpu_memory_vm_or_server, mock_detect_env
    ):
        """Test get_cpu_resources for VM environment."""
        logger = mock_logger()
        cpu_stats, env = get_cpu_resources(logger)
        self.assertEqual(cpu_stats, (16, 32768))
        self.assertEqual(env, "vm")
        logger.info.assert_any_call("Environment: vm")
        logger.info.assert_any_call("Available cores: 16")
        logger.info.assert_any_call("Available memory: 32.0 (GB)")

    @patch(
        "cellvit.utils.ressource_manager.detect_runtime_environment",
        return_value="server",
    )
    @patch(
        "cellvit.utils.ressource_manager.get_cpu_memory_vm_or_server",
        return_value=(32, 65536),
    )
    @patch("cellvit.utils.ressource_manager.NullLogger")
    def test_get_cpu_resources_server(
        self, mock_logger, mock_get_cpu_memory_vm_or_server, mock_detect_env
    ):
        """Test get_cpu_resources for server environment."""
        logger = mock_logger()
        cpu_stats, env = get_cpu_resources(logger)
        self.assertEqual(cpu_stats, (32, 65536))
        self.assertEqual(env, "server")
        logger.info.assert_any_call("Environment: server")
        logger.info.assert_any_call("Available cores: 32")
        logger.info.assert_any_call("Available memory: 64.0 (GB)")


class TestRessourceManagerGPU(TestCase):
    @patch("cellvit.utils.ressource_manager.NullLogger")
    @patch("torch.cuda.is_available", return_value=False)
    def test_get_gpu_resources_no_gpu(self, mock_is_available, mock_logger):
        """Test get_gpu_resources when no GPU is available."""
        logger = mock_logger()
        gpu_resources = get_gpu_resources(logger)
        self.assertFalse(gpu_resources["has_gpu"])
        self.assertEqual(gpu_resources["gpu_count"], 0)
        self.assertEqual(gpu_resources["details"], {})
        self.assertEqual(gpu_resources["devices"], {})
        logger.warning.assert_called_once_with("No CUDA-capable GPU detected.")

    @patch("cellvit.utils.ressource_manager.NullLogger")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.backends.cudnn.version", return_value=8000)
    @patch(
        "torch.cuda.get_device_name", side_effect=["GPU 0", "GPU 1", "GPU 2", "GPU 3"]
    )
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", side_effect=[1e9, 2e9])
    @patch("torch.cuda.memory_reserved", side_effect=[1.5e9, 2.5e9])
    def test_get_gpu_resources_with_gpus(
        self,
        mock_memory_reserved,
        mock_memory_allocated,
        mock_get_device_properties,
        mock_get_device_name,
        mock_cudnn_version,
        mock_device_count,
        mock_is_available,
        mock_logger,
    ):
        """Test get_gpu_resources with available GPUs."""
        # hint: mock more cpu devices than available
        logger = mock_logger()
        mock_get_device_properties.side_effect = [
            MagicMock(total_memory=8e9, major=7, minor=5),
            MagicMock(total_memory=8e9, major=7, minor=5),
            MagicMock(total_memory=8e9, major=7, minor=5),
            MagicMock(total_memory=8e9, major=7, minor=5),
        ]

        gpu_resources = get_gpu_resources(logger)

        self.assertTrue(gpu_resources["has_gpu"])
        self.assertEqual(gpu_resources["gpu_count"], 2)
        self.assertEqual(gpu_resources["details"]["cudnn_version"], 8000)

        self.assertIn(0, gpu_resources["devices"])
        self.assertIn(1, gpu_resources["devices"])

        device_0 = gpu_resources["devices"][0]
        self.assertEqual(device_0["name"], "GPU 0")
        self.assertEqual(device_0["total_memory_gb"], 8.0)
        self.assertEqual(device_0["compute_capability"], "7.5")

        device_1 = gpu_resources["devices"][1]
        self.assertEqual(device_1["name"], "GPU 1")
        self.assertEqual(device_1["total_memory_gb"], 8.0)
        self.assertEqual(device_1["compute_capability"], "7.5")

    @patch("cellvit.utils.ressource_manager.NullLogger")
    @patch("torch.cuda.is_available", side_effect=ImportError)
    def test_get_gpu_resources_pytorch_not_installed(
        self, mock_is_available, mock_logger
    ):
        """Test get_gpu_resources when PyTorch is not installed."""
        logger = mock_logger()
        gpu_resources = get_gpu_resources(logger)
        self.assertFalse(gpu_resources["has_gpu"])
        self.assertEqual(gpu_resources["gpu_count"], 0)
        self.assertEqual(gpu_resources["details"]["error"], "PyTorch not installed")
        logger.error.assert_called_once_with(
            "PyTorch not installed. Cannot check GPU availability."
        )

    @patch("cellvit.utils.ressource_manager.NullLogger")
    @patch("torch.cuda.is_available", side_effect=Exception("Unexpected error"))
    def test_get_gpu_resources_unexpected_error(self, mock_is_available, mock_logger):
        """Test get_gpu_resources with unexpected error."""
        logger = mock_logger()
        gpu_resources = get_gpu_resources(logger)
        self.assertFalse(gpu_resources["has_gpu"])
        self.assertEqual(gpu_resources["gpu_count"], 0)
        self.assertEqual(gpu_resources["details"]["error"], "Unexpected error")
        logger.error.assert_called_once_with(
            "Unexpected error during GPU check: Unexpected error"
        )


class TestGetUsedMemory(TestCase):
    @patch("cellvit.utils.ressource_manager.get_used_memory_slurm", return_value=1024.0)
    def test_get_used_memory_slurm(self, mock_get_used_memory_slurm):
        """Test get_used_memory for Slurm."""
        memory = get_used_memory("slurm")
        self.assertEqual(memory, 1024.0)
        mock_get_used_memory_slurm.assert_called_once()

    @patch(
        "cellvit.utils.ressource_manager.get_used_memory_kubernetes", return_value=512.0
    )
    def test_get_used_memory_kubernetes(self, mock_get_used_memory_kubernetes):
        """Test get_used_memory for Kubernetes."""
        memory = get_used_memory("kubernetes")
        self.assertEqual(memory, 512.0)
        mock_get_used_memory_kubernetes.assert_called_once()

    @patch("cellvit.utils.ressource_manager.get_used_memory_docker", return_value=256.0)
    def test_get_used_memory_docker(self, mock_get_used_memory_docker):
        """Test get_used_memory for Docker."""
        memory = get_used_memory("docker")
        self.assertEqual(memory, 256.0)
        mock_get_used_memory_docker.assert_called_once()

    @patch(
        "cellvit.utils.ressource_manager.get_used_memory_process", return_value=128.0
    )
    def test_get_used_memory_vm_or_server(self, mock_get_used_memory_process):
        """Test get_used_memory for VM or server."""
        memory = get_used_memory("vm")
        self.assertEqual(memory, 128.0)
        mock_get_used_memory_process.assert_called_once()

    @patch("psutil.Process")
    def test_get_used_memory_process(self, mock_process):
        """Test get_used_memory_process."""
        mock_process_instance = mock_process.return_value
        mock_process_instance.memory_info.return_value.rss = 50 * 1024 * 1024  # 50 MB
        mock_process_instance.children.return_value = []
        memory = get_used_memory_process()
        self.assertEqual(memory, 50.0)

    @patch(
        "os.path.exists",
        side_effect=lambda path: path == "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
    @patch("builtins.open", new_callable=mock_open, read_data="104857600")  # 100 MB
    def test_get_used_memory_kubernetes_with_cgroup(self, mock_open, mock_exists):
        """Test get_used_memory_kubernetes with cgroup."""
        memory = get_used_memory_kubernetes()
        self.assertEqual(memory, 100.0)

    @patch("os.path.exists", return_value=False)
    @patch("cellvit.utils.ressource_manager.get_used_memory_process", return_value=64.0)
    def test_get_used_memory_kubernetes_fallback(
        self, mock_get_used_memory_process, mock_exists
    ):
        """Test get_used_memory_kubernetes with fallback to process memory."""
        memory = get_used_memory_kubernetes()
        self.assertEqual(memory, 64.0)
        mock_get_used_memory_process.assert_called_once()

    @patch("os.environ", {"SLURM_JOB_ID": "12345"})
    @patch("subprocess.check_output", return_value="1024K")
    def test_get_used_memory_slurm_with_sstat(self, mock_subprocess):
        """Test get_used_memory_slurm with sstat."""
        memory = get_used_memory_slurm()
        self.assertEqual(memory, 1.0)  # 1024 KB = 1 MB
        mock_subprocess.assert_called_once()

    @patch("os.environ", {"SLURM_JOB_ID": "12345"})
    @patch("subprocess.check_output", side_effect=FileNotFoundError)
    @patch(
        "os.path.exists",
        side_effect=lambda path: path == "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
    @patch("builtins.open", new_callable=mock_open, read_data="209715200")  # 200 MB
    def test_get_used_memory_slurm_with_cgroup(
        self, mock_open, mock_exists, mock_subprocess
    ):
        """Test get_used_memory_slurm with cgroup."""
        memory = get_used_memory_slurm()
        self.assertEqual(memory, 200.0)

    @patch("os.environ", {"SLURM_JOB_ID": "12345"})
    @patch("subprocess.check_output", side_effect=FileNotFoundError)
    @patch("os.path.exists", return_value=False)
    @patch("cellvit.utils.ressource_manager.get_used_memory_process", return_value=32.0)
    def test_get_used_memory_slurm_fallback(
        self, mock_get_used_memory_process, mock_exists, mock_subprocess
    ):
        """Test get_used_memory_slurm with fallback to process memory."""
        memory = get_used_memory_slurm()
        self.assertEqual(memory, 32.0)
        mock_get_used_memory_process.assert_called_once()


class TestSystemConfiguration(unittest.TestCase):
    @patch("cellvit.utils.ressource_manager.get_cpu_resources")
    @patch("cellvit.utils.ressource_manager.get_gpu_resources")
    @patch("cellvit.utils.ressource_manager.check_module")
    @patch("cellvit.utils.ressource_manager.check_cupy")
    def test_initialization(
        self,
        mock_check_cupy,
        mock_check_module,
        mock_get_gpu_resources,
        mock_get_cpu_resources,
    ):
        # Mock CPU and GPU resources
        mock_get_cpu_resources.return_value = ((8, 16384), "docker")
        mock_get_gpu_resources.return_value = {
            "has_gpu": True,
            "gpu_count": 2,
            "devices": {
                0: {"total_memory_gb": 8.0},
                1: {"total_memory_gb": 16.0},
            },
        }
        mock_check_module.side_effect = lambda module: module in ["cupy", "ray"]
        mock_check_cupy.return_value = True

        # Initialize SystemConfiguration
        config = SystemConfiguration(gpu=1)

        # Assertions
        self.assertEqual(config.cpu_count, 8)
        self.assertEqual(config.memory, 16384)
        self.assertEqual(config.runtime_environment, "docker")
        self.assertTrue(config.has_gpu)
        self.assertEqual(config.gpu_count, 2)
        self.assertEqual(config.gpu_memory, 16.0)
        self.assertTrue(config.cupy)
        self.assertTrue(config.ray)
        self.assertFalse(config.cucim)
        self.assertFalse(config.numba)

    @patch("cellvit.utils.ressource_manager.get_cpu_resources")
    @patch("cellvit.utils.ressource_manager.get_gpu_resources")
    def test_invalid_gpu_index(self, mock_get_gpu_resources, mock_get_cpu_resources):
        # Mock CPU and GPU resources
        mock_get_cpu_resources.return_value = ((8, 16384), "docker")
        mock_get_gpu_resources.return_value = {
            "has_gpu": True,
            "gpu_count": 1,
            "devices": {
                0: {"total_memory_gb": 8.0},
            },
        }

        # Assert that initializing with an invalid GPU index raises an error
        with self.assertRaises(SystemError):
            SystemConfiguration(gpu=2)

    @patch("cellvit.utils.ressource_manager.logging.getLogger")
    @patch("cellvit.utils.ressource_manager.get_cpu_resources")
    @patch("cellvit.utils.ressource_manager.get_gpu_resources")
    def test_log_system_configuration(
        self, mock_get_gpu_resources, mock_get_cpu_resources, mock_get_logger
    ):
        # Mock CPU and GPU resources
        mock_get_cpu_resources.return_value = ((8, 16384), "docker")
        mock_get_gpu_resources.return_value = {
            "has_gpu": True,
            "gpu_count": 1,
            "devices": {
                0: {"total_memory_gb": 8.0},
            },
        }

        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Initialize SystemConfiguration
        config = SystemConfiguration()

        # Log system configuration
        config.log_system_configuration()

        # Assertions
        mock_logger.info.assert_any_call("========================================")
        mock_logger.info.assert_any_call("         SYSTEM CONFIGURATION           ")
        mock_logger.info.assert_any_call("========================================")
        mock_logger.info.assert_any_call(f"CPU count:          {config.cpu_count}")
        mock_logger.info.assert_any_call(
            f"Memory:             {config.memory / 1024:.2f} GB"
        )
        mock_logger.info.assert_any_call(f"GPU count:          {config.gpu_count}")
        mock_logger.info.assert_any_call(f"Used GPU-ID:        {config.gpu_index}")
        mock_logger.info.assert_any_call(
            f"GPU memory:         {config.gpu_memory:.2f} GB"
        )
        mock_logger.info.assert_any_call(f"Ray available:      {config.ray}")
        mock_logger.info.assert_any_call(f"Ray worker count:   {config.ray_worker}")
        mock_logger.info.assert_any_call(
            f"Ray remote cpus:    {config.ray_remote_cpus}"
        )
        mock_logger.info.assert_any_call(f"Cupy available:     {config.cupy}")
        mock_logger.info.assert_any_call(f"Cucim available:    {config.cucim}")
        mock_logger.info.assert_any_call(f"Numba available:    {config.numba}")
        mock_logger.info.assert_any_call("========================================")
        mock_logger.info.assert_any_call("       SYSTEM LOADED SUCCESSFULLY       ")
        mock_logger.info.assert_any_call("========================================")


if __name__ == "__main__":
    unittest.main()
