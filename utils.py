import torch
from random import randint
from helpers import o3d_knn, setup_camera, quat_mult, weighted_l2_loss_v2, l1_loss_v2
from PIL import Image
import copy
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from external import calc_ssim, build_rotation
import wandb
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

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


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data

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

def get_loss(params, batch, variables, alpha):
    rendervar = params2rendervar(params)
    # rendervar['means2D'].retain_grad()

    im, _, _, = Renderer(raster_settings=batch['cam'])(**rendervar)

    is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
    fg_pts = rendervar['means3D'][is_fg]
    fg_rot = rendervar['rotations'][is_fg]

    rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
    rot = build_rotation(rel_rot)
    neighbor_pts = fg_pts[variables["neighbor_indices"]]
    curr_offset = neighbor_pts - fg_pts[:, None]
    curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
    
    loss_rigid = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                            variables["neighbor_weight"])

    loss_l1 = torch.nn.functional.l1_loss(im, batch['im'])
    loss_ssim = (1.0 - calc_ssim(im, batch['im']))
    loss_im = 0.8*loss_l1 + 0.2*loss_ssim

    bg_pts = rendervar['means3D'][~is_fg]
    bg_rot = rendervar['rotations'][~is_fg]
    loss_bg = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

    wandb.log({
            f'loss-l1': loss_l1.item(),
            f'loss-ssim': loss_ssim.item(),
            f'loss-image': loss_im.item(),
            f'loss-rigid': loss_rigid.item(),
            f'loss-bg': loss_bg.item(),
        })

    return loss_im + 3*alpha*loss_rigid + 0*loss_bg

def init_variables(params, num_knn=20):
    variables = {}

    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means'][is_fg]
    init_bg_pts = params['means'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['rotations'][~is_fg])

    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)

    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['rotations']).detach()
    
    return variables

def initialize_per_timestep(params, variables):
    pts = params['means']
    rot = torch.nn.functional.normalize(params['rotations'])

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach().clone()
    variables['prev_offset'] = prev_offset.detach().clone()
    variables["prev_col"] = params['colors'].detach().clone()
    variables["prev_pts"] = pts.detach().clone()
    variables["prev_rot"] = rot.detach().clone()

    return variables

def get_frame(params, batch):
    rendervar = params2rendervar(params)

    im, _, _, = Renderer(raster_settings=batch['cam'])(**rendervar)

    return im

def load_params(path: str):
    params = torch.load(path)

    for v in params.values():
        v.requires_grad = False

    return params

def get_linear_warmup_cos_annealing(optimizer, warmup_iters, total_iters):
        scheduler_warmup = LinearLR(optimizer, start_factor=1/1000, total_iters=warmup_iters)
        scheduler_cos_decay = CosineAnnealingLR(optimizer, T_max=total_iters-warmup_iters)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, 
                                    scheduler_cos_decay], milestones=[warmup_iters])

        return scheduler