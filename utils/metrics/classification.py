import torch

def top1_accuracy(step_output_list):
    """
    Compute Top-1 accuracy across all batches by concatenating tensors.
    More efficient than per-batch accumulation.

    Args:
        step_output_list (list[dict]): Each dict must have keys
            - "pred": Tensor of shape (B, C)
            - "ground_truth": Tensor of shape (B,)

    Returns:
        float: Top-1 accuracy over the whole epoch.
    """
    if len(step_output_list) == 0:
        return 0.0

    preds = torch.cat([out["pred"].detach() for out in step_output_list], dim=0)
    labels = torch.cat([out["ground_truth"].detach() for out in step_output_list], dim=0)

    # Compute Top-1 predictions
    top1 = preds.argmax(dim=1)
    correct = float((top1 == labels).sum().item())
    total = float(labels.size(0))

    return correct / total


def top5_accuracy(step_output_list):
    """
    Compute Top-5 accuracy across all batches by concatenating tensors.
    More efficient than per-batch accumulation.

    Args:
        step_output_list (list[dict]): Each dict must have keys
            - "pred": Tensor of shape (B, C)
            - "ground_truth": Tensor of shape (B,)

    Returns:
        float: Top-5 accuracy over the whole epoch.
    """
    if len(step_output_list) == 0:
        return 0.0

    preds = torch.cat([out["pred"].detach() for out in step_output_list], dim=0)
    labels = torch.cat([out["ground_truth"].detach() for out in step_output_list], dim=0)

    # Compute Top-5 predictions
    top5 = preds.topk(5, dim=1).indices
    correct = float((top5 == labels.unsqueeze(1)).any(dim=1).sum().item())
    total = float(labels.size(0))

    return correct / total
