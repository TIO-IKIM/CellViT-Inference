python3 ./cellvit/detect_cells.py \
  --model SAM \
  --outdir ./test_results/x40_svs/folder/SAM \
  --cpu_count 16 \
  process_dataset \
  --wsi_folder ./test_database/x40_svs
