export NGPUS=8
/usr/bin/python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/SA_AutoAug/fcos_imprv_R_50_FPN_4x.yaml  OUTPUT_DIR models/fcos_imprv_R_50_FPN_4x_saautoaug
