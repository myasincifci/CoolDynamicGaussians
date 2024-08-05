import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, dim) -> None:
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x

        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        
        x += identity
        x = nn.functional.gelu(x)

        return x
    
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, seq_len, block_num) -> None:
        super(MLP, self).__init__()
        
        self.fc_in = nn.Linear(in_dim, hid_dim)
        self.res_blocks = nn.Sequential(
            *(ResidualBlock(hid_dim) for _ in range(block_num))
        )
        self.fc_out = nn.Linear(hid_dim, 7)

        self.emb = nn.Embedding(seq_len, 3)

    def forward(self, x, x_, t):
        identity = x

        x = torch.cat((x_, t), dim=1)

        x = self.fc_in(x)
        x = self.res_blocks(x)
        x = self.fc_out(x)

        x += identity

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, L):
        super(PositionalEncoding, self).__init__()
        self.L = L
        self.consts = ((torch.ones(L)*2).pow(torch.arange(L)) * torch.pi).cuda()
    
    def forward(self, x):
        x = x[:,:,None]
        A = (self.consts * x).repeat_interleave(2,2)
        A[:,:,::2] = torch.sin(A[:,:,::2])
        A[:,:,1::2] = torch.cos(A[:,:,::2])

        return A.permute(0,2,1).flatten(start_dim=1)