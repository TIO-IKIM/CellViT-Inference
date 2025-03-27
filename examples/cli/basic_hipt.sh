python3 ./cellvit/detect_cells.py \
  --model HIPT \
  --outdir ./test_results/x40_svs/minimal/HIPT \
  --cpu_count 16 \
  process_wsi \
  --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
