import torch.nn.functional as F


def calculate_action_loss(action_vals, action_logits):
    # TODO(jhoang): Calculate losses based on conditional commands?
    log_soft_max = F.log_softmax(action_logits, dim=3)
    # 1 + action_vals to make it positive
    return F.kl_div(log_soft_max, 1 + action_vals, reduction="none").mean(dim=[2, 3]).mean()
