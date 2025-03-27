python3 ./cellvit/detect_cells.py \
  --model SAM \
  --outdir ./test_results/x40_svs/batch_size/SAM \
  --batch_size 32 \
  --cpu_count 16 \
  --ray_worker 4 \
  --debug \
  process_dataset \
  --wsi_folder ./test_database/x40_svs
