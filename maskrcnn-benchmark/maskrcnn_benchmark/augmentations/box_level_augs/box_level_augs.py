import torch
import random
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.augmentations.box_level_augs.color_augs import color_aug_func
from maskrcnn_benchmark.augmentations.box_level_augs.geometric_augs import geometric_aug_func

pixel_mean = cfg.INPUT.PIXEL_MEAN


def _box_sample_prob(bbox, scale_ratios_splits, box_prob=0.3):
    scale_ratios, scale_splits = scale_ratios_splits

    ratios = np.array(scale_ratios)
    ratios = ratios / ratios.sum()
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area == 0:
        return 0
    if area < scale_splits[0]:
        scale_ratio = ratios[0]
    elif area < scale_splits[1]:
        scale_ratio = ratios[1]
    else:
        scale_ratio = ratios[2]
    return box_prob * scale_ratio


def _box_aug_per_img(img, target, aug_type=None, scale_ratios=None, scale_splits=None, img_prob=0.1, box_prob=0.3, level=1):
    if random.random() > img_prob:
        return img, target
    img_mean = torch.Tensor(pixel_mean).reshape(3,1,1).to(img.device)
    img += img_mean
    img /= 255.0

    tag = 'prob' if aug_type in geometric_aug_func else 'area'
    scale_ratios_splits = [scale_ratios[tag], scale_splits]
    if scale_ratios is None:
        box_sample_prob = [box_prob] * len(target.bbox)
    else:
        box_sample_prob = [_box_sample_prob(bbox, scale_ratios_splits, box_prob=box_prob) for bbox in target.bbox]

    if aug_type in color_aug_func:
        img_aug = color_aug_func[aug_type](img, level, target, [scale_ratios['area'], scale_splits], box_sample_prob)
    elif aug_type in geometric_aug_func:
        img_aug, target = geometric_aug_func[aug_type](img, level, target, box_sample_prob)
    else:
        raise ValueError('Unknown box-level augmentation function %s.' % (aug_type))

    out = img_aug*255.0-img_mean

    return out, target


class Box_augs(object):
    def __init__(self, box_augs_dict, max_iters, scale_splits, box_prob=0.3):
        self.max_iters = max_iters
        self.box_prob = box_prob
        self.scale_splits = scale_splits
        self.policies = box_augs_dict['policies']
        self.scale_ratios = box_augs_dict['scale_ratios']

    def __call__(self, tensor, target, iteration):
        iter_ratio = float(iteration) / self.max_iters
        sub_policy = random.choice(self.policies)
        tensor, _ = _box_aug_per_img(tensor, target, aug_type=sub_policy[0][0], scale_ratios=self.scale_ratios, scale_splits=self.scale_splits, img_prob=sub_policy[0][1] * iter_ratio, box_prob=self.box_prob, level=sub_policy[0][2])
        tensor_out, target_out = _box_aug_per_img(tensor, target, aug_type=sub_policy[1][0], scale_ratios=self.scale_ratios, scale_splits=self.scale_splits, img_prob=sub_policy[1][1] * iter_ratio, box_prob=self.box_prob, level=sub_policy[1][2])

        return tensor_out, target_out
