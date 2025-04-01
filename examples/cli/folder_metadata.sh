cellvit-inference \
    --model SAM \
    --outdir ./test_results/BRACS/folder/SAM \
    process_dataset \
    --wsi_folder ./test_database/BRACS \
    --wsi_extension tiff \
    --wsi_mpp 0.25 \
    --wsi_magnification 40
