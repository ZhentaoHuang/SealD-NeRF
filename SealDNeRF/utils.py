from nerf.utils import *
from nerf.utils import Trainer as _Trainer
from scipy.spatial.transform import Rotation
from typing import Union


class Trainer(_Trainer):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):

        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler

        super().__init__(name, opt, model, criterion, optimizer, ema_decay, lr_scheduler, metrics, local_rank, world_size, device, mute, fp16, eval_interval, max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboardX, scheduler_update_every_step)
        
    ### ------------------------------	

    def train_step(self, data, time_frame):

        if self.proxy_train:
            self.proxy_truth(data, time_frame, use_cache=False) #self.cache_gt

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        time = data['time'] # [B, 1]
        time = time_frame
        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, time, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            #torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        images = data['images'] # [B, N, 3/4]

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, time, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, **vars(self.opt))
    
        pred_rgb = outputs['image']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()

        # deform regularization
        # if 'deform' in outputs and outputs['deform'] is not None:
        #     loss = loss + 1e-3 * outputs['deform'].abs().mean()
        
        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        time = data['time'] # [B, 1]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, time, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        time = data['time'] # [B, 1]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, time, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, time=0, bg_color=None, spp=1, downscale=1, return_pos=False):
        # print("running test_gui in sealdnerf")
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'time': torch.FloatTensor([[time]]).to(self.device), # from scalar to [1, 1] tensor.
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=spp)
                # self.print_character()
        preds_depth = torch.nan_to_num(preds_depth)
        if self.ema is not None:
            self.ema.restore()
        

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        if return_pos:
            # unnorm_d = rays['rays_d'].view(H, W, 3) * rays['rays_d_norm'].view(H, W, 1)
            # preds_pos = rays['rays_o'].view(H, W, 3) + preds_depth.view(H, W, 1) * unnorm_d
            preds_pos = rays['rays_o'].view(H, W, 3) + preds_depth.view(H, W, 1) * rays['rays_d'].view(H, W, 3)
            # preds_pos = preds_pos.view(H, W, 3)
            preds_pos = preds_pos.detach().cpu().numpy()
            # cv2.imwrite("pos.exr", preds_pos)
            # cv2.imwrite("rays_o.exr", rays['rays_o'].view(H, W, 3).detach().cpu().numpy())
            # cv2.imwrite("rays_d_norm.exr", rays['rays_d_norm'].view(H, W, 1).detach().cpu().numpy())
            # cv2.imwrite("rays_d.exr", rays['rays_d'].view(H, W, 3).detach().cpu().numpy())
            # cv2.imwrite("rays_d_unnorm.exr", unnorm_d.detach().cpu().numpy())

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }
        if return_pos:
            outputs['pos'] = preds_pos

        return outputs        

    def save_mesh(self, time, save_path=None, resolution=256, threshold=10):
        # time: scalar in [0, 1]
        time = torch.FloatTensor([[time]]).to(self.device)

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device), time)['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    def print_character(self):
        print("Teacher model")



class StudentTrainer(Trainer):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 student_model, teacher_model, # network 
                 proxy_train = True,
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        self.teacher_model = teacher_model
        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        self.proxy_train = proxy_train

        super().__init__(name, opt, student_model, criterion, optimizer, ema_decay, lr_scheduler, metrics, local_rank, world_size, device, mute, fp16, eval_interval, max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboardX, scheduler_update_every_step)
        
    def print_character(self):
        print("Student model")


    def sample_points(self, bounds: torch.Tensor, point_step=0.005, angle_step=45):
        """
        Sample points per step inside bounds (B, 2, 3) or (2, 3)
        """
        if bounds.ndim == 2:
            bounds = bounds[None]
        sampled_points = []
        sampled_dirs = []
        for i in range(bounds.shape[0]):
            coords_min, coords_max = bounds[i]
            X, Y, Z = torch.meshgrid(torch.arange(coords_min[0], coords_max[0], step=point_step),
                                    torch.arange(
                coords_min[1], coords_max[1], step=point_step),
                torch.arange(coords_min[2], coords_max[2], step=point_step))
            sampled_points.append(torch.stack(
                [X, Y, Z], dim=-1).reshape(-1, 3))

            r_x, r_y, r_z = torch.meshgrid(torch.arange(0, 360, step=angle_step),
                                        torch.arange(0, 360, step=angle_step),
                                        torch.arange(0, 360, step=angle_step))
            eulers = torch.stack([r_x, r_y, r_z], dim=-1).reshape(-1, 3)
            sampled_dirs.append(torch.from_numpy(Rotation.from_euler('xyz', eulers.numpy(
            ), degrees=True).apply(np.array([1-1e-5, 0, 0]))))

        # trimesh.PointCloud(
        #     self.sampled_points.cpu().numpy()).export('tmp/sampled.obj')
        return torch.concat(sampled_points), torch.concat(sampled_dirs)


    def freeze_module(self, module: Union[torch.nn.ParameterList, torch.nn.ModuleList, torch.nn.Module], freeze: bool):
        module.training = not freeze
        if isinstance(module, (torch.nn.ParameterList, torch.nn.ModuleList)):
            for i in range(len(module)):
                module[i].requires_grad_(not freeze)
        elif isinstance(module, torch.nn.Module):
            module.requires_grad_(not freeze)


    def freeze_mlp_deform(self, freeze: bool = True):
            """
            freeze all MLPs or unfreeze them by passing `freeze=False`
            """
            # if self._backbone is BackBoneTypes.TensoRF:
            #     # freeze_module(self.model.color_net, freeze)
            #     # freeze_module(self.model.sigma_mat, freeze)
            #     # freeze_module(self.model.sigma_vec, freeze)
            #     # freeze_module(self.model.color_mat, freeze)
            #     # freeze_module(self.model.color_vec, freeze)
            #     # freeze_module(self.model.basis_mat, freeze)
            #     return
            # elif self._backbone is BackBoneTypes.NGP:
            self.freeze_module(self.model.deform_net, freeze)
            # self.freeze_module(self.model.color_net, freeze)
            # if hasattr(self.model, 'bg_net') and self.model.bg_net is not None:
            #     self.freeze_module(self.model.bg_net, freeze)


    def freeze_mlp(self, freeze: bool = True):
        """
        freeze all MLPs or unfreeze them by passing `freeze=False`
        """
        # if self._backbone is BackBoneTypes.TensoRF:
        #     # freeze_module(self.model.color_net, freeze)
        #     # freeze_module(self.model.sigma_mat, freeze)
        #     # freeze_module(self.model.sigma_vec, freeze)
        #     # freeze_module(self.model.color_mat, freeze)
        #     # freeze_module(self.model.color_vec, freeze)
        #     # freeze_module(self.model.basis_mat, freeze)
        #     return
        # elif self._backbone is BackBoneTypes.NGP:
        self.freeze_module(self.model.sigma_net, freeze)
        self.freeze_module(self.model.color_net, freeze)
        if hasattr(self.model, 'bg_net') and self.model.bg_net is not None:
            self.freeze_module(self.model.bg_net, freeze)




    def init_pretraining(self, time_frame, epochs=0, batch_size=4096, lr=0.07,
                     local_point_step=0.001, local_angle_step=45,
                     surrounding_point_step=0.01, surrounding_angle_step=45, surrounding_bounds_extend=0.2,
                     global_point_step=0.05, global_angle_step=45, no_debug: bool = False):
        """
        call this until seal_mapper is initialized
        """
        time_frame = torch.FloatTensor([[time_frame]]).to(self.device)
        # pretrain epochs before the real training starts
        self.pretraining_epochs = epochs
        self.pretraining_batch_size = batch_size
        self.pretraining_lr = lr
        if self.pretraining_epochs > 0:
            # simply use L1 to compute pretraining loss
            self.pretraining_criterion = torch.nn.L1Loss().to(self.device)
            # sample points and dirs from seal mapper
            self.pretraining_data = {}

            t = time.time()
            # prepare local data and gt
            if local_point_step > 0:
                local_bounds = self.teacher_model.seal_mapper.map_data['force_fill_bound']
                local_points, local_dirs = self.sample_points(
                    local_bounds, local_point_step, local_angle_step)
                local_points = local_points.to(
                    self.device, torch.float32)
                local_dirs = local_dirs.to(
                    self.device, torch.float32)

                # map sampled points
                mapped_points, mapped_dirs, mapped_mask = self.teacher_model.seal_mapper.map_to_origin(local_points, torch.zeros_like(
                    local_points, device=self.device, dtype=torch.float32) + torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32))
                # if we enable map source, all points inside fill bound should be kept
                if 'map_source' in self.teacher_model.seal_mapper.map_data:
                    mapped_mask[:] = True
                # filter sampled points. only store masked ones
                local_points = local_points[mapped_mask]
                N_local_points = local_points.shape[0]
                # prepare sampled dirs so we won't need to do randomly sampling in the tringing time
                local_dirs = local_dirs[torch.randint(
                    local_dirs.shape[0], (N_local_points,), device=self.device)]

                # infer gt sigma & color from teacher model and store them
                mapped_points = mapped_points[mapped_mask]
                mapped_dirs = mapped_dirs[mapped_mask]

                if hasattr(self.teacher_model, 'secondary_teacher_model'):
                    gt_sigma, gt_color = self.teacher_model.secondary_teacher_model(
                        mapped_points, mapped_dirs)
                else:
                    gt_sigma, gt_color = self.teacher_model(
                        mapped_points, mapped_dirs, time_frame)

                # map gt color
                gt_color = self.teacher_model.seal_mapper.map_color(
                    mapped_points, mapped_dirs, gt_color)

                # prepare pretraining steps to avoid cuda oom
                local_steps = list(
                    range(0, N_local_points, self.pretraining_batch_size))
                if local_steps[-1] != N_local_points:
                    local_steps.append(N_local_points)

                self.pretraining_data['local'] = {
                    'points': local_points,
                    'dirs': local_dirs,
                    'sigma':  gt_sigma.detach(),
                    'color': gt_color.detach(),
                    'steps': local_steps
                }
                self.is_pretraining = True

            print(f"Local x generation: {time.time()-t}")
            t = time.time()

            # prepare surrounding data and gt
            if surrounding_point_step > 0:
                # (B, 2, 3) or (2, 3)
                surrounding_bounds: torch.Tensor = self.teacher_model.seal_mapper.map_data[
                    'force_fill_bound']
                if surrounding_bounds.ndim == 2:
                    surrounding_bounds[0] -= surrounding_bounds_extend
                    surrounding_bounds[0] = torch.max(
                        surrounding_bounds[0], self.model.aabb_train[:3])
                    surrounding_bounds[1] += surrounding_bounds_extend
                    surrounding_bounds[1] = torch.min(
                        surrounding_bounds[1], self.model.aabb_train[3:])
                else:
                    surrounding_bounds[:, 0] -= surrounding_bounds_extend
                    surrounding_bounds[:, 0] = torch.max(
                        surrounding_bounds[:, 0], self.model.aabb_train[:3])
                    surrounding_bounds[:, 1] += surrounding_bounds_extend
                    surrounding_bounds[:, 1] = torch.min(
                        surrounding_bounds[:, 1], self.model.aabb_train[3:])
                surrounding_points, surrounding_dirs = self.sample_points(
                    surrounding_bounds, surrounding_point_step, surrounding_angle_step)
                surrounding_points = surrounding_points.to(
                    self.device, torch.float32)
                surrounding_dirs = surrounding_dirs.to(
                    self.device, torch.float32)

                # map sampled points
                _, _, mapped_mask = self.teacher_model.seal_mapper.map_to_origin(surrounding_points, torch.zeros_like(
                    surrounding_points, device=self.device, dtype=torch.float32) + torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32))
                # filter sampled points. only store unmasked ones
                surrounding_points = surrounding_points[~mapped_mask]
                N_surrounding_points = surrounding_points.shape[0]
                # prepare sampled dirs so we won't need to do randomly sampling in the tringing time
                surrounding_dirs = surrounding_dirs[torch.randint(
                    surrounding_dirs.shape[0], (N_surrounding_points,), device=self.device)]

                gt_sigma, gt_color = self.teacher_model(
                    surrounding_points, surrounding_dirs, time_frame)

                # prepare pretraining steps to avoid cuda oom
                surrounding_steps = list(
                    range(0, N_surrounding_points, self.pretraining_batch_size))
                if surrounding_steps[-1] != N_surrounding_points:
                    surrounding_steps.append(N_surrounding_points)

                self.pretraining_data['surrounding'] = {
                    'points': surrounding_points,
                    'dirs': surrounding_dirs,
                    'sigma':  gt_sigma.detach(),
                    'color': gt_color.detach(),
                    'steps': surrounding_steps
                }

            print(f"Surrounding x generation: {time.time()-t}")
            t = time.time()

            # prepare global data and gt
            if global_point_step > 0:
                global_bounds = self.model.aabb_train.view(2, 3)
                global_points, global_dirs = self.sample_points(
                    global_bounds, global_point_step, global_angle_step)
                global_points = global_points.to(
                    self.device, torch.float32)
                global_dirs = global_dirs.to(
                    self.device, torch.float32)

                _, _, mapped_mask = self.teacher_model.seal_mapper.map_to_origin(global_points, torch.zeros_like(
                    global_points, device=self.device, dtype=torch.float32) + torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32))

                # keep non-edited points
                global_points = global_points[~mapped_mask]
                N_global_points = global_points.shape[0]
                global_dirs = global_dirs[torch.randint(
                    global_dirs.shape[0], (N_global_points,), device=self.device)]

                gt_sigma, gt_color = self.teacher_model(
                    global_points, global_dirs)

                # prepare pretraining steps to avoid cuda oom
                global_steps = list(
                    range(0, N_global_points, self.pretraining_batch_size))
                if global_steps[-1] != N_global_points:
                    global_steps.append(N_global_points)

                self.pretraining_data['global'] = {
                    'points': global_points,
                    'dirs': global_dirs,
                    'sigma':  gt_sigma.detach(),
                    'color': gt_color.detach(),
                    'steps': global_steps
                }

            print(f"Global x generation: {time.time()-t}")
            t = time.time()

            if not no_debug:
                visualize_dir = os.path.join(self.workspace, 'pretrain_vis')
                if not os.path.exists(visualize_dir):
                    os.makedirs(visualize_dir)
                for k, v in self.pretraining_data.items():
                    trimesh.PointCloud(v['points'].view(-1, 3).cpu().numpy(), v['color'].view(-1, 3).cpu().numpy()).export(
                        os.path.join(visualize_dir, f'{k}.ply'))

    def set_lr(self, lr: float):
        """
        manually set learning rate to speedup pretraining. restore the original lr by passing `lr=-1`
        """
        if lr < 0:
            if not hasattr(self, '_cached_lr') or self._cached_lr is None:
                return
            lr = self._cached_lr
            self._cached_lr = None
        else:
            self._cached_lr = self.optimizer.param_groups[0]['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @torch.no_grad()
    def proxy_truth(self, data, time_frame, all_ray: bool = True, use_cache: bool = False, n_batch: int = 1):

        time = data['time'] # [B, 1]
        time = time_frame
        """
        proxy the ground truth RGB from teacher model
        """
        # already proxied the dataset
        if 'skip_proxy' in data and data['skip_proxy']:
            return
        # avoid OOM
        torch.cuda.empty_cache()
        # if the model's bitfield is not hacked, do it before inferring
        if not self.teacher_model.density_bitfield_hacked:
            self.teacher_model.hack_bitfield()

        # if we want a full image (B, H, W, C) or just rays (B, N, C)
        is_full = False
        if 'images' in data:
            images = data['images']  # [B, N, 3/4]
            is_full = images.ndim == 4
            image_shape = images.shape
        elif 'images_shape' in data:
            image_shape = data['images_shape']
            is_full = len(image_shape) == 4

        # if use_cache, the training can be slower. might be useful with very small dataset
        use_cache = use_cache and 'pixel_index' in data and not is_full
        is_skip_computing = False

        if use_cache:
            compute_mask = ~self.proxy_cache_mask[data['data_index'],
                                                data['pixel_index']]
            is_skip_computing = not compute_mask.any()

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        if use_cache:
            rays_o = rays_o[compute_mask][None]
            rays_d = rays_d[compute_mask][None]

        if not is_skip_computing:
            teacher_outputs = {
                'image': [],
                'depth': []
            }
            with torch.no_grad():
                total_batches = rays_o.shape[1]
                batch_size = total_batches // n_batch
                if (total_batches % n_batch):
                    n_batch += 1
                for i in range(n_batch):
                    current_teacher_outputs = self.teacher_model.render(
                        rays_o[:, i*batch_size:(i+1)*batch_size, :], rays_d[:, i*batch_size:(i+1)*batch_size, :], time,staged=True, bg_color=None, perturb=False, force_all_rays=all_ray, **vars(self.opt))
                    teacher_outputs['image'].append(
                        current_teacher_outputs['image'])
                    teacher_outputs['depth'].append(
                        current_teacher_outputs['depth'])
                teacher_outputs['image'] = torch.concat(
                    teacher_outputs['image'], 1)
                teacher_outputs['depth'] = torch.concat(
                    teacher_outputs['depth'], 1)

        if use_cache:
            if not is_skip_computing:
                self.proxy_cache_image[data['data_index'], data['pixel_index']
                                    [compute_mask]] = torch.nan_to_num(teacher_outputs['image'], nan=0.).detach()
                self.proxy_cache_depth[data['data_index'], data['pixel_index']
                                    [compute_mask]] = torch.nan_to_num(teacher_outputs['depth'], nan=0.).detach()
                self.proxy_cache_mask[data['data_index'], data['pixel_index']
                                    [compute_mask]] = True
            data['images'] = self.proxy_cache_image[data['data_index'],
                                                    data['pixel_index']].to(self.device)
            data['depths'] = self.proxy_cache_depth[data['data_index'],
                                                    data['pixel_index']].to(self.device)
        else:
            data['images'] = torch.nan_to_num(teacher_outputs['image'], nan=0.)
            data['depths'] = torch.nan_to_num(teacher_outputs['depth'], nan=0.)
        # reshape if it is a full image
        if is_full:
            data['images'] = data['images'].view(*image_shape[:-1], -1)
            data['depths'] = data['depths'].view(*image_shape[:-1], -1)





    def train_gui(self, time_frame,train_loader, step=16, is_pretraining=False):

        # print(is_pretraining)
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()

        # mark untrained grid
        # if self.global_step == 0:
        #     self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        if is_pretraining:
            st = time.time()
            torch.cuda.synchronize()
            for _ in range(step):
                self.pretrain_one_epoch(True)
            ed = time.time()
            torch.cuda.synchronize()
            self.log(f"[INFO]Pretraining epoch x{step} time: {ed-st:.4f}s")
            return {
                'loss': 0.0,
                'lr': self.optimizer.param_groups[0]['lr']
            }

        self.freeze_mlp(False)
        self.freeze_mlp_deform(True)
        self.set_lr(-1)

        # if the model's bitfield is not hacked, do it before inferring
        if not self.teacher_model.density_bitfield_hacked:
            # time_frame = torch.FloatTensor([[time_frame]]).to(self.device)
             
            
            # time_frame = math.floor(time_frame * 128)
            # self.teacher_model.time_frame = time_frame
            self.teacher_model.hack_bitfield()
        # if not self.has_proxied:
        #     # if hasattr(self, 'is_proxy_running'):
        #     #     if self.is_proxy_running.value:
        #     #         return {
        #     #             'loss': 0.0,
        #     #             'lr': self.optimizer.param_groups[0]['lr']
        #     #         }
        #     #     else:
        #     #         self.has_proxied = True
        #     # self.is_proxy_running = Value(ctypes.c_bool, True)
        #     train_loader.extra_info['provider'].proxy_dataset(
        #         self.teacher_model, n_batch=1)
        #     self.has_proxied = True
        #     # return {
        #     #     'loss': 0.0,
        #     #     'lr': self.optimizer.param_groups[0]['lr']
        #     # }

        loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            time_frame = torch.FloatTensor([[time_frame]]).to(self.device)

            # t1 = torch.floor(time_frame[0][0] * 128).clamp(min=0, max=128 - 1).long()
            # t2 = torch.floor(data["time"][0][0] * 128).clamp(min=0, max=128 - 1).long()
            # # print(t1, t2)
            # if(t1 != t2):
            #     continue
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data, time_frame)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs



    # def trainer_constructor(base, backbone: BackBoneTypes):
    #     """
    #     Construct trainer class with dynamically selected base class
    #     """
    #     Trainer = type(f'Trainer_{backbone.name}', (base,), {
    #         '__init__': init,
    #         'init_pretraining': init_pretraining,
    #         'train': train,
    #         'train_step': train_step,
    #         'eval_step': eval_step,
    #         'test_step': test_step,
    #         'pretrain_one_epoch': pretrain_one_epoch,
    #         'pretrain_part': pretrain_part,
    #         'pretrain_step': pretrain_step,
    #         'freeze_mlp': freeze_mlp,
    #         'set_lr': set_lr,
    #         'proxy_truth': proxy_truth,
    #         'train_gui': train_gui,
    #         '_backbone': backbone
    #     })
    #     Trainer._self = Trainer
    #     return Trainer

    # def init(self: trainer_types, name, opt, student_model, teacher_model, proxy_train=True, proxy_test=False, proxy_eval=False, cache_gt=False, criterion=None, optimizer=None, ema_decay=None, lr_scheduler=None, metrics=..., local_rank=0, world_size=1, device=None, mute=False, fp16=False, eval_interval=1, eval_count=None, max_keep_ckpt=2, workspace='workspace', best_mode='min', use_loss_as_metric=True, report_metric_at_train=False, use_checkpoint="latest", use_tensorboardX=True, scheduler_update_every_step=False):
    #     super(self._self, self).__init__(name, opt, student_model, criterion=criterion, optimizer=optimizer, ema_decay=ema_decay, lr_scheduler=lr_scheduler, metrics=metrics, local_rank=local_rank, world_size=world_size, device=device, mute=mute, fp16=fp16, eval_interval=eval_interval, eval_count=eval_count,
    #                                     max_keep_ckpt=max_keep_ckpt, workspace=workspace, best_mode=best_mode, use_loss_as_metric=use_loss_as_metric, report_metric_at_train=report_metric_at_train, use_checkpoint=use_checkpoint, use_tensorboardX=use_tensorboardX, scheduler_update_every_step=scheduler_update_every_step)
    #     # use teacher trainer instead of teacher model directly
    #     # to make sure it's properly initialized, e.g. device
    #     self.teacher_model = teacher_model

    #     # flags indicating the proxy behavior of different stages
    #     self.proxy_train = proxy_train
    #     self.proxy_eval = proxy_eval
    #     self.proxy_test = proxy_test
    #     self.is_pretraining = False
    #     self.has_proxied = False

    #     self.cache_gt = cache_gt