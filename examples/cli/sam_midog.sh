cellvit-inference \
    --model SAM \
    --nuclei_taxonomy midog \
    --outdir ./test_results/BRACS/midog/SAM \
    process_wsi \
    --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
    --wsi_mpp 0.25 \
    --wsi_magnification 40
