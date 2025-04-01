cellvit-inference \
    --model SAM \
    --outdir ./test_results/x40_svs/cpu_count_ray_worker/SAM \
    --cpu_count 16 \
    --ray_worker 4 \
    process_dataset \
    --wsi_folder ./test_database/x40_svs
