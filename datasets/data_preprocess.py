import torch


def Mean_std_process(inputs):
    inputs = (inputs - torch.mean(inputs, dim=-1).unsqueeze(-1)) / torch.std(inputs, dim=-1).unsqueeze(-1)

    return inputs


def Mix_max_process(inputs):

    inputs = ((inputs - torch.min(inputs, dim=-1)[0].unsqueeze(-1)) /
              (torch.max(inputs, dim=-1)[0].unsqueeze(-1) - torch.min(inputs, dim=-1)[0].unsqueeze(-1))) * 2 - 1

    return inputs