import torch


def scale_jitter(tensor, target, jitter_factor):
    if isinstance(jitter_factor, tuple):
        new_h, new_w = jitter_factor
    elif isinstance(jitter_factor, float):
        _, h, w = tensor.shape
        new_h, new_w = int(h * jitter_factor), int(w * jitter_factor)
    else:
        return tensor, target
    
    tensor_out = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='nearest').squeeze(0)
    target_out = target.resize((new_w, new_h))
    return tensor_out, target_out
