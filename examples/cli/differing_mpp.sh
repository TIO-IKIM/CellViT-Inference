python3 ./cellvit/detect_cells.py \
  --model HIPT \
  --outdir ./test_results/x20_svs/minimal/HIPT \
  --cpu_count 16 \
  process_wsi \
  --wsi_path ./test_database/x20_svs/CMU-1-Small-Region.svs
