import math
import torch
import random
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.augmentations.image_level_augs.scale_jitter import scale_jitter


class Zoom_out(object):
    def __init__(self, ratio=1.0, img_pool_size=100, iou_threshold=0.5, size_divisible=2):
        self.ratio = ratio
        self.img_pool = []
        self.img_pool_size = img_pool_size
        self.iou_threshold =iou_threshold
        self.size_divisible = size_divisible

    def __call__(self, tensor, target):
        if self.ratio >= 1.0:
            return tensor, target

        self.img_pool.append({'tensor': tensor, 'target': target})

        if len(self.img_pool) > self.img_pool_size:
            self.img_pool.pop(0)

        c, h, w = tensor.shape
        if self.size_divisible > 0:
            h = int(math.ceil(h / self.size_divisible) * self.size_divisible)
            w = int(math.ceil(w / self.size_divisible) * self.size_divisible)

        new_h, new_w = int(self.ratio * h), int(self.ratio * w)
        in_tensor, in_target = scale_jitter(tensor, target, (new_h, new_w))

        if len(self.img_pool) < 4:
            tensor_out, target_out = in_tensor, in_target
        else:
            pad_imgs = random.sample(self.img_pool, 3)
            pad_tensors, pad_targets = [], []
            for img in pad_imgs:
                pad_tensor, pad_target = scale_jitter(img['tensor'], img['target'], (new_h, new_w))
                pad_tensors.append(pad_tensor)
                pad_targets.append(pad_target)

            crop_boxes = [(0, 0, w-new_w, new_h), (0, 0, new_w, h-new_h), (0, 0, w-new_w, h-new_h)]

            tensor_out = in_tensor.new(*(c, h, w)).zero_()
            tensor_out[:c, :new_h, :new_w].copy_(in_tensor)
            tensor_out[:c, :new_h, new_w:].copy_(pad_tensors[0][:c, :crop_boxes[0][3], :crop_boxes[0][2]])
            tensor_out[:c, new_h:, :new_w].copy_(pad_tensors[1][:c, :crop_boxes[1][3], :crop_boxes[1][2]])
            tensor_out[:c, new_h:, new_w:].copy_(pad_tensors[2][:c, :crop_boxes[2][3], :crop_boxes[2][2]])

            crop_targets = []
            for i, pad_target in enumerate(pad_targets):
                crop_target = pad_target.crop(crop_boxes[i])
                ious = crop_target.area() / pad_target.area()
                crop_target = crop_target[ious >= self.iou_threshold]
                crop_targets.append(crop_target)

            offsets_box = [torch.Tensor([0.0,0.0,0.0,0.0]), torch.Tensor([new_w, 0.0, new_w, 0.0]), torch.Tensor([0.0, new_h, 0.0, new_h]), torch.Tensor([new_w, new_h, new_w, new_h])]
            target_out = torch.cat([target.bbox + offsets_box[i] for i, target in enumerate([in_target] + crop_targets)], dim=0)
            target_out = BoxList(target_out, (w, h), mode='xyxy')
            target_out.add_field('labels', torch.cat([target.extra_fields['labels'] for target in ([in_target] + crop_targets)], dim=-1))

            polys_list = [[poly.polygons[0] for poly in target.extra_fields['masks'].instances.polygons] for target in ([in_target] + crop_targets)]
            offsets_mask = [[0.0, 0.0], [new_w, 0.0], [0.0, new_h], [new_w, new_h]]

            syn_mask = []
            for i, polys in enumerate(polys_list):
                syn_mask += [[list(np.array(poly)+np.array(offsets_mask[i]*int(len(poly)/2)))] for poly in polys]

            syn_mask = SegmentationMask(syn_mask, (w, h), mode='poly')
            target_out.add_field('masks', syn_mask)

        return tensor_out, target_out
