cellvit-inference \
    --model SAM \
    --outdir ./test_results/x40_svs/cpu_count/SAM \
    --cpu_count 16 \
    process_dataset \
    --wsi_folder ./test_database/x40_svs
