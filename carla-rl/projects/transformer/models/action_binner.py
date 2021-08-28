import numpy as np
import torch


ACTIONS = torch.FloatTensor([
    [-1, 1/3],  [-1, 2/3], [-1, 1],
    [-.75,1/3], [-.75,2/3],[-.75,1],
    [-.5,1/3],  [-.5,2/3], [-.5,1],
    [-.25,1/3], [-.25,2/3],[-.25,1],
    [0, 1/3],   [0, 2/3],  [0, 1],
    [.25,1/3], [.25,2/3],[.25,1],
    [.5,1/3],  [.5,2/3], [.5,1],
    [.75,1/3], [.75,2/3],[.75,1],
    [1, 1/3],  [1, 2/3], [1, 1],
    [0, -1]
]).reshape(28,2)


def action_tokenizer(actions, expand_dim=True, pad_to_size=128):
    dists = torch.cdist(actions, ACTIONS)
    action_ids = dists.argmin(dim=-1)
    if expand_dim:
        action_ids = torch.nn.functional.one_hot(action_ids, num_classes=pad_to_size)
    return action_ids


def decode_action_tokens(action_tokens):
    print(action_tokens)
    action_tokens = action_tokens.clip(0, 27).long()
    actions = ACTIONS[action_tokens]
    return actions
