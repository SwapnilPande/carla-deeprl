import torch.nn.functional as F

TEMPERATURE = 0.01


def calculate_action_loss(action_vals, action_logits):
    act_probs = F.softmax(action_vals / TEMPERATURE, dim=3)
    # TODO(jhoang): Calculate losses based on conditional commands?
    log_soft_max = F.log_softmax(action_logits, dim=3)
    return F.kl_div(log_soft_max, act_probs, reduction="none").mean(dim=[2, 3]).mean()
