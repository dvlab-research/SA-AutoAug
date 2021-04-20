export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/SA_AutoAug/atss_dcnv2_X_101_32x8d_FPN_2x.yaml  OUTPUT_DIR models/atss_dcnv2_X_101_32x8d_FPN_2x_saautoaug
