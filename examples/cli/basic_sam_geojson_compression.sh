python3 ./cellvit/detect_cells.py \
  --model SAM \
  --outdir ./test_results/x40_svs/compression_graph_geojson/SAM \
  --geojson \
  --compression \
  --graph \
  --cpu_count 16 \
  process_wsi \
  --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
