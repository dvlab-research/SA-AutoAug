import torch
import numpy as np
from fcos_core.augmentations.image_level_augs.scale_jitter import scale_jitter


class Zoom_in(object):
    def __init__(self, ratio=1.0, iou_threshold=0.5):
        self.ratio = ratio
        self.iou_threshold = iou_threshold

    def __call__(self, tensor, target):
        if self.ratio <= 1.0:
            return tensor, target

        h, w = tensor.shape[1], tensor.shape[2]
        new_h, new_w = int(h * self.ratio), int(w * self.ratio)

        original_target = target.copy_with_fields(list(target.extra_fields.keys()))
        enlarged_tensor, enlarged_target = scale_jitter(tensor, target, (new_h, new_w))
        original_enlarged_target = enlarged_target.copy_with_fields(list(enlarged_target.extra_fields.keys()))

        crop_x, crop_y = np.random.randint(0, new_h - h), np.random.randint(0, new_w - w)
        crop_box = (crop_y, crop_x, crop_y + w, crop_x + h)
        cropped_tensor = enlarged_tensor[:, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        cropped_target = enlarged_target.crop(crop_box)
        ious = cropped_target.area() / original_enlarged_target.area()
        cropped_target = cropped_target[ious >= self.iou_threshold]

        if len(cropped_target) > 0:
            tensor_out = cropped_tensor
            target_out = cropped_target
        else:
            tensor_out = tensor
            target_out = original_target

        return tensor_out, target_out
