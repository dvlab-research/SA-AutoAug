export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/search.py --config-file configs/SA_AutoAug/retinanet_R-50-FPN_search.yaml OUTPUT_DIR /path/to/searchlog_dir