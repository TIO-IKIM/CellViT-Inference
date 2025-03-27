python3 ./cellvit/detect_cells.py \
  --model SAM \
  --outdir ./test_results/x40_svs/debug/SAM \
  --cpu_count 16 \
  --debug \
  process_dataset \
  --wsi_folder ./test_database/x40_svs
