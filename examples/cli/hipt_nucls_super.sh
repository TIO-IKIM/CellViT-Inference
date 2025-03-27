python3 ./cellvit/detect_cells.py \
  --model HIPT \
  --nuclei_taxonomy nucls_super \
  --outdir ./test_results/BRACS/nucls_super/HIPT \
  --cpu_count 16 \
  process_wsi \
  --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
  --wsi_mpp 0.25 \
  --wsi_magnification 40
