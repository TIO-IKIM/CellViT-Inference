python3 ./cellvit/detect_cells.py \
  --model SAM \
  --outdir ./test_results/BRACS/folder/SAM \
  --cpu_count 16 \
  process_dataset \
  --wsi_folder ./test_database/BRACS \
  --wsi_extension tiff \
  --wsi_mpp 0.25 \
  --wsi_magnification 40

python3 ./cellvit/detect_cells.py \
                --model SAM \
                --outdir ./test_results/Philips/folder/SAM \
                  --cpu_count 16 \
                process_dataset \
                --wsi_folder ./test_database/Philips \
                --wsi_extension tiff
