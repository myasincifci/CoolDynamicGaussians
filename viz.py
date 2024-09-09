import sys
import json
import torch
from torch import nn
import numpy as np
from helpers import setup_camera
from PIL import Image
import copy
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import quat_mult, weighted_l2_loss_v2, l1_loss_v2
from external import calc_ssim, build_rotation

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

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, seq_len, block_num) -> None:
        super(MLP, self).__init__()
        
        self.fc_in = nn.Linear(in_dim, hid_dim)
        self.res_blocks = nn.Sequential(
            *(ResidualBlock(hid_dim) for _ in range(block_num))
        )
        self.fc_out = nn.Linear(hid_dim, 7)

        # self.emb = nn.Embedding(seq_len, 3)

    def dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def forward(self, x, x_, t):
        B, D = x.shape

        identity = x

        # e = self.emb(t).repeat(B, 1)
        # e = self.dec2bin(t, 3).repeat(B, 1)

        x = torch.cat((x_, t), dim=1)
        # x = x + e

        x = self.fc_in(x)
        x = self.res_blocks(x)
        x = self.fc_out(x)

        x += identity

        return x

def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset

def load_params(path: str):
    params = torch.load(path)

    for v in params.values():
        v.requires_grad = False

    return params

def params2rendervar(params):
    rendervar = {
        'means3D': params['means'],
        'colors_precomp': params['colors'],
        'rotations': torch.nn.functional.normalize(params['rotations']),
        'opacities': torch.sigmoid(params['opacities']),
        'scales': torch.exp(params['scales']),
        'means2D': torch.zeros_like(params['means'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

def get_image_loss(params, batch):
    rendervar = params2rendervar(params)
    # rendervar['means2D'].retain_grad()

    im, _, _, = Renderer(raster_settings=batch['cam'])(**rendervar)
    
    loss_l1 = torch.nn.functional.l1_loss(im, batch['im'])
    loss_ssim = (1.0 - calc_ssim(im, batch['im']))
    loss_im = 0.8*loss_l1 + 0.2*loss_ssim

    return loss_im

def get_frame(params, batch):
    rendervar = params2rendervar(params)

    im, _, _, = Renderer(raster_settings=batch['cam'])(**rendervar)

    return im

def main():
    sequence = sys.argv[1]
    init_cloud = sys.argv[2]
    T = int(sys.argv[3])
    weights_path = sys.argv[4]

    md = json.load(open(f"./data/{sequence}/train_meta.json", 'r'))
    params = load_params(init_cloud)

    mlp = MLP(100, 128, T, 6).cuda()

    means = params['means']
    rotations = params['rotations']

    means_norm = means - means.min(dim=0).values
    means_norm = (2. * means_norm / means_norm.max(dim=0).values) - 1.

    rotations_norm = rotations - rotations.min(dim=0).values
    rotations_norm = (2. * rotations_norm / rotations_norm.max(dim=0).values) - 1.

    pos_mean = PositionalEncoding(L=10)
    pos_smol = PositionalEncoding(L=4)

    means_norm = pos_mean(means_norm)
    rotations_norm = pos_smol(rotations_norm)

    # dataset = []
    # for t in range(1, T + 1, 1):
    #     dataset += [get_dataset(t, md, sequence)]

    # l = 0.01
    # with torch.no_grad():    
    #     for i, d in enumerate(dataset):
    #             sample_losses = []
    #             for s in d:
    #                 t = pos_smol(torch.tensor((i+1)/149).view(1,1).repeat(means_norm.shape[0], 1).cuda())

    #                 delta = mlp(torch.cat((params['means'], params['rotations']), dim=1), torch.cat((means_norm, rotations_norm), dim=1), t)
    #                 delta_means = delta[:,:3]
    #                 delta_rotations = delta[:,3:]

    #                 updated_params = copy.deepcopy(params)
    #                 updated_params['means'] = updated_params['means'].detach()
    #                 updated_params['means'] += delta_means * l
    #                 updated_params['rotations'] = updated_params['rotations'].detach()
    #                 updated_params['rotations'] += delta_rotations * l


    #                 loss = get_image_loss(updated_params, s)
    #                 sample_losses.append(loss.item())
    #             print(sum(sample_losses)/len(sample_losses))

    ## Visualize
    with torch.no_grad():
        ds = get_dataset(0, md=md, seq=sequence)
        canon_batch = ds[0]

        frames = []

        # Canonical frame
        frames.append(get_frame(params, canon_batch))

        for t_ in range(1, T+1 , 1):
            das = get_dataset(t_, md=md, seq=sequence)
            X = das[0]

            t = pos_smol(torch.tensor((t_)/149).view(1,1).repeat(means_norm.shape[0], 1).cuda())

            # delta = mlp(torch.cat((params['means'], params['rotations']), dim=1), torch.tensor(t).cuda())
            delta = mlp(torch.cat((params['means'], params['rotations']), dim=1), torch.cat((means_norm, rotations_norm), dim=1), t)
            delta_means = delta[:,:3]
            delta_rotations = delta[:,3:]

            updated_params = copy.deepcopy(params)
            updated_params['means'] = updated_params['means'].detach()
            updated_params['means'] += delta_means * l
            updated_params['rotations'] = updated_params['rotations'].detach()
            updated_params['rotations'] += delta_rotations * l

            fr = get_frame(updated_params, X)

            frames.append(fr)

        for i, frame in enumerate(frames):
            frame_np = (frame.detach().cpu().clip(min=0.0, max=1.0).permute(1,2,0).numpy()*255).astype(np.uint8)
            im = Image.fromarray(frame_np)
            im.save(f'image-{i}.png')

if __name__ == '__main__':
    main()