import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from .utils import custom_meshgrid

from dnerf.renderer import NeRFRenderer, sample_pdf
from SealNeRF.seal_utils import get_seal_mapper #temporarily


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()

class SealNeRFRenderer(NeRFRenderer):
    def __init__(self, bound=1, cuda_ray=False, density_scale=1, min_near=0.2, density_thresh=0.01, bg_radius=-1, **kwargs):
        super().__init__(bound=bound, cuda_ray=cuda_ray, density_scale=density_scale,
                         min_near=min_near, density_thresh=density_thresh, bg_radius=bg_radius)
        self.seal_mapper = None
        # Setting bitfield preciously
        # force_fill_bitfield_offsets: np.ndarray = self.force_fill_grid_indices % 8
        # self.force_fill_bitfield_values = torch.rand(
        #     [self.force_fill_bitfield_indices.shape[0], 1], dtype=torch.uint8, device=self.density_bitfield.device)
        # self.force_fill_bitfield_values = 255
        self.density_bitfield_origin = None
        self.density_bitfield_hacked = False

    def init_mapper(self, config_dir: str = '', config_dict: dict = None, config_file: str = 'seal.json', mapper = None):
        print("config_dir", config_dir)
        print("config_dict", len(config_dict))
        print("config_file", config_file)
        if mapper is None:
            self.seal_mapper = get_seal_mapper(config_dir, config_dict, config_file)
        else:
            self.seal_mapper = mapper
        print("seal_mapper", self.seal_mapper)
        if(self.seal_mapper is not None):
            print("seal_mapper not none")
        # B, 2, 3 or 2, 3
        bounds: torch.Tensor = self.seal_mapper.map_data['force_fill_bound']
        if bounds.ndim == 2:
            bounds = bounds[None]

        bounds[:, 0, :] = torch.max(bounds[:, 0, :], self.aabb_infer[:3].to(bounds.device))
        bounds[:, 1, :] = torch.min(bounds[:, 1, :], self.aabb_infer[-3:].to(bounds.device))
        grid_indices = []
        bitfield_indices = []
        for i in range(bounds.shape[0]):
            coords_min, coords_max = torch.floor(
                ((bounds[i] + self.bound) / self.bound / 2) * self.grid_size)
            # print(coords_min, coords_max)
            X, Y, Z = torch.meshgrid(torch.arange(coords_min[0], coords_max[0]),
                                    torch.arange(coords_min[1], coords_max[1]),
                                    torch.arange(coords_min[2], coords_max[2]))
            coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
            current_grid_indices = raymarching.morton3D(
                coords).long()  # [N]
            current_bitfield_indices = current_grid_indices // 8
            grid_indices.append(current_grid_indices)
            bitfield_indices.append(current_bitfield_indices)
        self.force_fill_grid_indices = torch.concat(grid_indices)
        self.force_fill_bitfield_indices = torch.concat(bitfield_indices)

    def update_extra_state(self, decay=0.95, S=128):
        # self.force_fill_grids()
        super().update_extra_state(decay, S)
        if self.seal_mapper is not None:
            self.hack_bitfield()

    @torch.no_grad()
    def hack_grids(self):
        self.density_grid[:,
                          self.force_fill_grid_indices] = min(self.mean_density * 1.5, self.density_thresh) + 1e-5

    @torch.no_grad()
    def hack_bitfield(self):
       
        print("origin",self.density_bitfield_origin)
        # if self.density_bitfield_origin is None:
        #     print("Going into hackbit", self.density_bitfield[0][self.force_fill_bitfield_indices])
        #     self.density_bitfield_origin = self.density_bitfield[0][self.force_fill_bitfield_indices]
            # for t in range(self.time_size):
            #     self.density_bitfield[t][self.force_fill_bitfield_indices] = 255
        self.density_bitfield_hacked = True
        # self.density_bitfield[:,
        #                       self.force_fill_bitfield_indices] = torch.bitwise_or(self.density_bitfield[:,
        #                                                                                                  self.force_fill_bitfield_indices], self.force_fill_bitfield_values)

    @torch.no_grad()
    def restore_bitfield(self):
        for t in range(self.time_size):
            self.density_bitfield[t][self.force_fill_bitfield_indices] = self.density_bitfield_origin
        self.density_bitfield_hacked = False


class SealNeRFTeacherRenderer(SealNeRFRenderer):
    def __init__(self, bound=1, cuda_ray=False, density_scale=1, min_near=0.2, density_thresh=0.01, bg_radius=-1, log2_hashmap_size=18, **kwargs):
        super().__init__(bound=bound, cuda_ray=cuda_ray, log2_hashmap_size=log2_hashmap_size,
                         density_scale=density_scale, min_near=min_near, density_thresh=density_thresh, bg_radius=bg_radius)


    def run_cuda(self, rays_o, rays_d, time, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(
            rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(
                rays_o, rays_d, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d)  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        # determine the correct frame of density grid to use
        t = torch.floor(time[0][0] * self.time_size).clamp(min=0, max=self.time_size - 1).long()
        self.time_frame = t
        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()  # set to 0
            self.local_step += 1

            # set to density_bitfield[t]
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(
                rays_o, rays_d, self.bound, self.density_bitfield[t], self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

            #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
            # print("self.seal_mapper",self.seal_mapper)
            
            if self.seal_mapper is not None:
                mapped_xyzs, mapped_dirs, mapped_mask = self.seal_mapper.map_to_origin(
                    xyzs.view(-1, 3), dirs.view(-1, 3))
            else:
                mapped_xyzs, mapped_dirs = xyzs, dirs

            # trimesh.PointCloud(
            #     xyzs.reshape(-1, 3).detach().cpu().numpy()).export('tmp/xyzs.obj')
            # trimesh.PointCloud(
            #     mapped_xyzs.reshape(-1, 3).detach().cpu().numpy()).export('tmp/xyzs_mapped.obj')

            sigmas, rgbs, deform = self(mapped_xyzs.view(xyzs.shape),
                                mapped_dirs.view(dirs.shape), time)

            # for across-model editing. non-mapped points are inferred from self, while mapped points are inferred from self.secondary_teacher_model
            # if hasattr(self, 'secondary_teacher_model'):
            #     secondary_sigmas, secondary_rgbs = self.secondary_teacher_model(mapped_xyzs.view(xyzs.shape)[mapped_mask], mapped_dirs.view(dirs.shape)[mapped_mask])
            #     sigmas[mapped_mask] = secondary_sigmas
            #     rgbs[mapped_mask] = secondary_rgbs
            # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
            # sigmas = density_outputs['sigma']
            # rgbs = self.color(xyzs, dirs, **density_outputs)
            sigmas = self.density_scale * sigmas
          
            # TODO zhentao, temporaly delete
            # if self.seal_mapper is not None:
                # rgbs[mapped_mask] = self.seal_mapper.map_color(mapped_xyzs[mapped_mask], mapped_dirs[mapped_mask], rgbs[mapped_mask])

            #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            # special case for CCNeRF's residual learning
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(
                        sigmas[k], rgbs[k], deltas, rays, T_thresh)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    # depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))

                depth = torch.stack(depths, axis=0)  # [K, B, N]
                image = torch.stack(images, axis=0)  # [K, B, N, 3]

            else:

                weights_sum, depth, image = raymarching.composite_rays_train(
                    sigmas, rgbs, deltas, rays, T_thresh)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                # depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)

            results['deform'] = deform
            results['weights_sum'] = weights_sum

        else:

            # allocate outputs
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(
                n_alive, dtype=torch.int32, device=device)  # [N]
            rays_t = nears.clone()  # [N]

            step = 0

            while step < max_steps:

                # count alive rays
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield[t],
                                                            self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)


                # print("243", self.seal_mapper)
                if self.seal_mapper is not None:
                    # print("line 244")
                    mapped_xyzs, mapped_dirs, mapped_mask = self.seal_mapper.map_to_origin(
                        xyzs.view(-1, 3), dirs.view(-1, 3))
                    # print("mapped_mask",mapped_mask)
                else:
                    mapped_xyzs, mapped_dirs = xyzs, dirs
                # print("251", self.seal_mapper)
                sigmas, rgbs, _ = self(mapped_xyzs.view(
                    xyzs.shape), mapped_dirs.view(dirs.shape), time)
                # if hasattr(self, 'secondary_teacher_model'):
                #     secondary_sigmas, secondary_rgbs = self.secondary_teacher_model(mapped_xyzs.view(xyzs.shape)[mapped_mask], mapped_dirs.view(dirs.shape)[mapped_mask])
                #     sigmas[mapped_mask] = secondary_sigmas
                #     rgbs[mapped_mask] = secondary_rgbs
                # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
                # sigmas = density_outputs['sigma']
                # rgbs = self.color(xyzs, dirs, **density_outputs)
                sigmas = self.density_scale * sigmas

                # print("self.seal_mapper",self.seal_mapper)
                # print("264", self.seal_mapper)
                if self.seal_mapper is not None:
                    rgbs[mapped_mask] = self.seal_mapper.map_color(mapped_xyzs[mapped_mask], mapped_dirs[mapped_mask], rgbs[mapped_mask]).to(rgbs.dtype)

                raymarching.composite_rays(
                    n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            # depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)

        results['depth'] = depth
        results['image'] = image

        return results


class SealNeRFStudentRenderder(SealNeRFRenderer):
    def __init__(self, bound=1, cuda_ray=False, density_scale=1, min_near=0.2, density_thresh=0.01, bg_radius=-1, log2_hashmap_size=18, **kwargs):
        super().__init__(bound=bound, cuda_ray=cuda_ray, log2_hashmap_size=log2_hashmap_size,
                         density_scale=density_scale, min_near=min_near, density_thresh=density_thresh, bg_radius=bg_radius)
