import torch.nn.functional as F

def cross_entropy_loss(pred, gt):
    return F.cross_entropy(pred, gt)