python3 ./cellvit/detect_cells.py \
  --model SAM \
  --outdir ./test_results/BRACS/folder/SAM \
  --cpu_count 16 \
  process_dataset \
  --wsi_folder ./test_database/BRACS \
  --wsi_mpp 0.25 \
  --wsi_magnification 40
