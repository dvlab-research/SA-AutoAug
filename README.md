# SA-AutoAug
**Scale-aware Automatic Augmentation for Object Detection**

Yukang Chen<sup>*</sup>, Yanwei Li<sup>*</sup>, Tao Kong, Lu Qi, Ruihang Chu, Lei Li, Jiaya Jia

<!-- [[`Paper`](https://arxiv.org/abs/2103.17220)] [[`BibTeX`](#CitingSAAutoAug)] -->

<div align="center">
  <img src="docs/Framework.png"/>
</div><br/>

This project provides the implementation for the CVPR 2021 paper "[Scale-aware Automatic Augmentation for Object Detection](https://arxiv.org/pdf/2103.17220.pdf)".
Scale-aware AutoAug provides a new search space and search metric to find effective data agumentation policies for object detection.
It is implemented on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [FCOS](https://github.com/tianzhi0549/FCOS). Both search and training codes have been released.
 To facilitate more use, we re-implement the training code based on [Detectron2](https://github.com/facebookresearch/detectron2). 



## Installation
For [maskrcnn-benchmark](https://github.com/Jia-Research-Lab/SA-AutoAug/tree/master/maskrcnn-benchmark) code, please follow [INSTALL.md](https://github.com/Jia-Research-Lab/SA-AutoAug/tree/master/maskrcnn-benchmark/INSTALL.md) for instruction.

For [FCOS](https://github.com/Jia-Research-Lab/SA-AutoAug/tree/master/FCOS) code, please follow [INSTALL.md](https://github.com/Jia-Research-Lab/SA-AutoAug/tree/master/FCOS/INSTALL.md) for instruction.

For [Detectron2](https://github.com/Jia-Research-Lab/SA-AutoAug/tree/master/detectron2), please follow [INSTALL.md](https://github.com/Jia-Research-Lab/SA-AutoAug/tree/master/detectron2/INSTALL.md) for instruction.

## Search
(You can skip this step and directly train on our searched policies.)

To search with 8 GPUs, run:
```bash
cd /path/to/SA-AutoAug/maskrcnn-benchmark
export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/search.py --config-file configs/SA_AutoAug/retinanet_R-50-FPN_search.yaml OURPUT_DIR /path/to/searchlog_dir
```

Since we finetune on an existing baseline model during search, a baseline model is needed. 
You can download this [model](https://drive.google.com/file/d/1jUbN6NIfabKEXB5CNTXMaORbGlzNYzuV/view?usp=sharing) for search, or you can use other Retinanet baseline model trained by yourself.

## Training
To train the searched policies on maskrcnn-benchmark (FCOS)
```bash
cd /path/to/SA-AutoAug/maskrcnn-benchmark
export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/SA_AutoAug/CONFIG_FILE  OUTPUT_DIR /path/to/traininglog_dir
```

For example, to train the retinanet ResNet-50 model with our searched data augmentation policies in 6x schedule:
```bash
cd /path/to/SA-AutoAug/maskrcnn-benchmark
export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/SA_AutoAug/retinanet_R-50-FPN_6x.yaml  OUTPUT_DIR models/retinanet_R-50-FPN_6x_SAAutoAug
```

To train the searched policies on detectron2
```bash
cd /path/to/SA-AutoAug/detectron2
python3 ./tools/train_net.py --num-gpus 8 --config-file ./configs/COCO-Detection/SA_AutoAug/CONFIG_FILE OUTPUT_DIR /path/to/traininglog_dir
```

For example, to train the retinanet ResNet-50 model with our searched data augmentation policies in 6x schedule:
```bash
cd /path/to/SA-AutoAug/detectron2
python3 ./tools/train_net.py --num-gpus 8 --config-file ./configs/COCO-Detection/SA_AutoAug/retinanet_R_50_FPN_6x.yaml OUTPUT_DIR output_retinanet_R_50_FPN_6x_SAAutoAug
```


## Results
We provide the results on COCO *val2017* set with pretrained models.

|  Method   | Backbone  | AP | Model | 
|  ----  | ----  | ----  | ----  |
| Faster R-CNN  | ResNet-50 | 41.8 | [Model](https://drive.google.com/file/d/1TdIKVfCwyiSmpRQcksi1ISIDgl-EjqSV/view?usp=sharing) |
| Faster R-CNN  | ResNet-101 | 44.2 | [Model](https://drive.google.com/file/d/1VDlHqR9mKD-ZfnyzaOd_eyemo97e3KdH/view?usp=sharing) |
| RetinaNet  | ResNet-50 | 41.4 | [Model](https://drive.google.com/file/d/1ojtT1eIcEhIiRo1OZZmT2QBriSan9U7b/view?usp=sharing) |
| RetinaNet  | ResNet-101 | 42.8 | [Model](https://drive.google.com/file/d/19mYsWpeMBLvIpdhXRYGYKX6C_63PFhld/view?usp=sharing) |
| Mask R-CNN  | ResNet-50 | 38.1 | [Model](https://drive.google.com/file/d/1DdacDkXs-lZ4iMutsONvKbPuDmwxpg9h/view?usp=sharing) |
| Mask R-CNN  | ResNet-101 | 40.0 | [Model](https://drive.google.com/file/d/1qi7G39CyLzeYnsmsIXcOM8ZigVehq3O0/view?usp=sharing) |
| FCOS  | ResNet-50 | 42.6 | [Model](https://drive.google.com/file/d/12QECU5eRwmoM461ci2yk4MuQ74TiiCp6/view?usp=sharing) |
| FCOS  | ResNet-101 | 44.0 | [Model](https://drive.google.com/file/d/1dEvERXupNwYsGZZ2V2H9eeM5wjwsbpPr/view?usp=sharing) |
| ATSS  | ResNext-101-32x8d-dcnv2 | 48.5 | [Model](https://drive.google.com/file/d/12_EnIO0sazi2HWMSChr15gnZFpFpXtK0/view?usp=sharing) |
| ATSS  | ResNext-101-32x8d-dcnv2 | 49.6 | [Model](https://drive.google.com/file/d/1wWyOI2udwPWBeM5Plk4XBxNPFdgixam0/view?usp=sharing) |


|  Method   | Backbone  | AP | Model | 
|  ----  | ----  | ----  | ----  | 
| Faster R-CNN  | ResNet-50 | 41.9 | [model](https://drive.google.com/file/d/1jgxnw1-b4ZnTNyn9rR_7u6vkvqGR1ks3/view?usp=sharing) [metrics](https://drive.google.com/file/d/16d1MyFVPWHJK__O0FQcAwxG8uWc8vAhi/view?usp=sharing) |
| Faster R-CNN  | ResNet-101 | 44.2 | [model](https://drive.google.com/file/d/10A16hUKpL2ffNpk38cOq5V9GuDB7OwwS/view?usp=sharing) [metrics](https://drive.google.com/file/d/1LAAD06iJ3vG7AwMjg9mHwTBPqvGXte9p/view?usp=sharing) |
| RetinaNet  | ResNet-50 | 40.8 | [model](https://drive.google.com/file/d/1GHAhrBa-TV_tJp3HGmWO02gZF06XP2vF/view?usp=sharing) [metrics](https://drive.google.com/file/d/15P05HgmXC1-Id-9LMtLb11fkAyBD3_Yy/view?usp=sharing) |
| RetinaNet  | ResNet-101 | 42.9 | [model](https://drive.google.com/file/d/1zYPTVvu-KnOSKzXxiEvSaav2OqcomqqF/view?usp=sharing) [metrics](https://drive.google.com/file/d/1_8QYMlJvvuEf35cty7SXrN-Hg1IHQVxh/view?usp=sharing) |
| Mask R-CNN  | ResNet-50 | - | Training |
| Mask R-CNN  | ResNet-101 | - | Training |

## Citing SA-AutoAug

Consider cite SA-Autoaug in your publications if it helps your research.

```
@inproceedings{qi2021msad,
  title={Scale-aware Automatic Augmentation for Object Detection},
  author={Yukang Chen, Yanwei Li, Tao Kong, Lu Qi, Ruihang Chu, Lei Li, Jiaya Jia},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## Acknowledgments
This training code of this project is built on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), [Detectron2](https://github.com/facebookresearch/detectron2), [FCOS](https://github.com/tianzhi0549/FCOS), and [ATSS](https://github.com/sfzhang15/ATSS). The search code of this project is modified from [DetNAS](https://github.com/megvii-model/DetNAS). Some augmentation code and settings follow [AutoAug-Det](https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/autoaugment_utils.py). We thanks a lot for the authors of these projects.

Note that:

(1) We also provides script files for search and training in [maskrcnn-benchmark](https://github.com/Jia-Research-Lab/SA-AutoAug/tree/master/maskrcnn-benchmark), [FCOS](https://github.com/Jia-Research-Lab/SA-AutoAug/tree/master/FCOS), and, [detectron2](https://github.com/Jia-Research-Lab/SA-AutoAug/tree/master/detectron2).

(2) Any issues or pull requests on this project are welcome. In addition, if you meet problems when tring to apply the augmentations on other datasets or codebase, feel free to contact Yukang Chen (yukangchen@cse.cuhk.edu.hk).

