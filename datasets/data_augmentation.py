import torch


def mixup(inputs):
    Beta = torch.distributions.beta.Beta(1, 1)
    lam = Beta.sample().to(inputs.device)
    index = torch.randperm(inputs.shape[0])
    x_a, x_b = inputs, inputs[index]
    inputs = lam * x_a + (1 - lam) * x_b
    return inputs, index, lam

def cutmix(inputs):
    index = torch.randperm(inputs.shape[0])
    x_a, x_b = inputs, inputs[index]
    Beta = torch.distributions.beta.Beta(1, 1)
    lam = Beta.sample().to(inputs.device)
    r_x = (torch.rand(1) * inputs.shape[-1]).to(inputs.device)
    r_w = 1 - lam
    x_1 = torch.clip(r_x - r_w / 2, min=0).to(torch.int)
    x_2 = torch.clip(r_x + r_w / 2, max=inputs.shape[-1]).to(torch.int)
    x_a[:, :, x_1:x_2] = x_b[:, :, x_1:x_2]
    inputs = x_a
    return inputs, index, lam