
import torch.nn.functional as F


def calculate_segmentation_loss(img, pred_img):
    return F.cross_entropy(F.interpolate(pred_img, scale_factor=4), img)
