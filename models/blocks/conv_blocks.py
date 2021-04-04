import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, ks=3, s=1, p=1, d=1, g=1, b=False,
            batch_norm=True, activation=True):
        super().__init__()
        operations = [nn.Conv2d(in_c, out_c, kernel_size=ks, stride=s, padding=p, 
               dilation=d, groups=g, bias=b)]
        if batch_norm:
            operations.append(nn.BatchNorm2d(out_c))
        if activation:
            operations.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*operations)

    def forward(self, x):
        return self.block(x)
