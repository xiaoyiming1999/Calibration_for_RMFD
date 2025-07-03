import torch
import torch.nn.functional as F
from torch import nn

class DropBlock1D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock1D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size


    def forward(self, x):
        # shape: (bsize, channels, width)

        # get gamma value
        gamma = self._compute_gamma()

        # sample mask
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        # place mask on input device
        mask = mask.to(x.device)

        # compute block mask
        block_mask = self._compute_block_mask(mask)

        # apply block mask
        out = x * block_mask[:, None, :]

        # scale output
        out = out * block_mask.numel() / block_mask.sum()

        return out


    def _compute_gamma(self):
        return self.drop_prob / self.block_size


    def _compute_block_mask(self, mask):
        block_mask = F.max_pool1d(input=mask[:, None, :],
                                  kernel_size=self.block_size,
                                  stride=1,
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask


if __name__ == '__main__':
    model = DropBlock1D(drop_prob=0.2, block_size=16)
    model.eval()
    input = torch.randn(size=[2, 1, 10])
    output_1 = model(input)
    output_2 = model(input)
    print(output_1)
    print(output_2)


