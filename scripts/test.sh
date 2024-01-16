python3 main_SealNeRF.py data/nerf_synthetic/lego/\
    --workspace exps/lego_ngp_bbox -O --bound 1.0 --scale 0.8 --dt_gamma 0\
    --pretraining_epochs 100 --extra_epochs 150\
    --pretraining_local_point_step 0.005 --pretraining_surrounding_point_step -1\
    --pretraining_lr 0.05 --ckpt exps/lego_ngp_new/checkpoints/ngp_ep0300.pth\
    --teacher_workspace exps/lego_ngp --teacher_ckpt exps/lego_ngp_new/checkpoints/ngp_ep0300.pth\
    --seal_config data/seal/lego_bbox/\
    --eval_interval 100 --eval_count 10 --gui


python main_seald.py data/dnerf/lego --workspace dnerf-jumpingjackslayer5-1103-2-t128 -O --bound 1.0 --scale 0.8 --dt_gamma 0 --lr 5e-4 --lr_net 5e-5 --gui --teacher_workspace trial_seald_teacher


python main_dnerf.py data/dnerf/standup --workspace exps/standup/layer8-t64-1206 --bound 1.0 --scale 0.8 --dt_gamma 0  --lr 5e-4 --lr_net 5e-4 -O --iters 300000

