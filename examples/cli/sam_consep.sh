cellvit-inference \
    --model SAM \
    --nuclei_taxonomy consep \
    --outdir ./test_results/BRACS/consep/SAM \
    process_wsi \
    --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
    --wsi_mpp 0.25 \
    --wsi_magnification 40
