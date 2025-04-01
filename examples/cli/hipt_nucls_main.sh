cellvit-inference \
    --model HIPT \
    --nuclei_taxonomy nucls_main \
    --outdir ./test_results/BRACS/nucls_main/HIPT \
    process_wsi \
    --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
    --wsi_mpp 0.25 \
    --wsi_magnification 40
