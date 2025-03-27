python3 ./cellvit/detect_cells.py \
  --model HIPT \
  --outdir ./test_results/all/HIPT \
  --cpu_count 16 \
  process_dataset \
  --wsi_filelist ./test_database/filelist.csv
