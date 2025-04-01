cellvit-inference \
    --model SAM \
    --outdir ./test_results/x40_svs/ray_worker_cpu_remote/SAM \
    --cpu_count 16 \
    --ray_worker 6 \
    --ray_remote_cpus 1 \
    process_dataset \
    --wsi_folder ./test_database/x40_svs
