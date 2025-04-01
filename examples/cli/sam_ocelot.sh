cellvit-inference \
    --model SAM \
    --nuclei_taxonomy ocelot \
    --outdir ./test_results/BRACS/ocelot/SAM \
    process_wsi \
    --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
    --wsi_mpp 0.25 \
    --wsi_magnification 40
