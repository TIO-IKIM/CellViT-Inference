python3 ./cellvit/detect_cells.py \
  --model SAM \
  --outdir ./test-results/x_40/minimal \
  --cpu_count 16 \
  process_wsi \
  --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs

python3 ./cellvit/detect_cells.py \
  --model HIPT \
  --outdir ./test-results/x_40/full_geojson \
  --geojson \
  --cpu_count 16 \
  process_wsi \
  --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs

python3 ./cellvit/detect_cells.py \
  --model SAM \
  --outdir ./test-results/x_40/compression \
  --geojson \
  --compression \
  --graph \
  --cpu_count 16 \
  process_wsi \
  --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs

python3 ./cellvit/detect_cells.py \
  --model HIPT \
  --outdir ./test-results/x20/1 \
  process_wsi \
  --wsi_path ./test_database/x20_svs/CMU-1-Small-Region.svs

python3 ./cellvit/detect_cells.py \
  --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --outdir ./test-results/x20/2 \
  --geojson \
  process_wsi \
  --wsi_path ./test_database/x20_svs/CMU-1-Small-Region.svs \
  --wsi_properties "{\"slide_mpp\": 0.50}"

python3 ./cellvit/detect_cells.py \
  --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --outdir ./test-results/x_40/minimal \
  process_wsi \
  --wsi_path ./test_database/MIDOG/001_pyramid.tiff
  --wsi_properties "{\"slide_mpp\": 0.25, \"magnification\": 40}"

python3 ./cellvit/detect_cells.py \
  --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --outdir ./test-results/filelist \
  --geojson \
  process_dataset \
  --filelist ./test_database/MIDOG/example_filelist.csv

  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/MIDOG/filelist \
    process_dataset \
    --filelist ./test_database/MIDOG/example_filelist.csv

    python3 ./cellvit/detect_cells.py \
      --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
      --outdir ./test-results/x20/binary \
      --binary \
      --geojson \
      process_wsi \

      --wsi_path ./test_database/x20_svs/CMU-1-Small-Region.svs


  bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/x_40/minimal \
    --classifier_path ./checkpoints/classifier/sam-h/consep.pth
    process_wsi \
    --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs



python3 ./cellvit/detect_cells.py \
  --model SAM \
  --outdir ./test_database_2/Phillips/output \
  --cpu_count 16 \
  --nuclei_taxonomy panoptils \
  --geojson \
  process_wsi \
  --wsi_path ./test_database_2/Phillips/input/Philips-1.tiff \
  --wsi_magnification 40
