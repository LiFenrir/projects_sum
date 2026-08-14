import torch
import torch.nn.functional as F



def dice_loss(
    pred,
    target,
    eps=1e-6
):

    pred = torch.sigmoid(pred)


    pred = pred.flatten(1)

    target = target.flatten(1)


    inter = (
        pred * target
    ).sum(1)


    union = (
        pred.sum(1)
        +
        target.sum(1)
    )


    loss = (
        1 -
        (
            2*inter + eps
        )
        /
        (
            union + eps
        )
    )


    return loss.mean()



def segmentation_loss(
    pred,
    target
):

    bce = F.binary_cross_entropy_with_logits(
        pred,
        target
    )


    dice = dice_loss(
        pred,
        target
    )


    return (
        0.5*bce
        +
        0.5*dice
    )