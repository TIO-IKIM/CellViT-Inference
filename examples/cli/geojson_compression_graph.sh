cellvit-inference \
    --model SAM \
    --outdir ./test_results/x40_svs/compression_graph_geojson/SAM \
    --geojson \
    --graph \
    --compression \
    process_wsi \
    --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
