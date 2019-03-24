import torch


def dice_coef(preds, truth, smooth=1.0):
    """Dice coefficient for PyTorch."""
    return (2*(preds*truth).sum() + smooth) / ((preds+truth).sum() + smooth)
