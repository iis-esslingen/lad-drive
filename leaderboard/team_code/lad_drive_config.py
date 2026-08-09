import os


class GlobalConfig:
    """base architecture configurations"""

    # Controller
    turn_KP = 1.2
    turn_KI = 0.8
    turn_KD = 0.35
    turn_n = 40  # buffer size
    
    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller
    preception_model = 'memfuser_baseline_e1d3_return_feature'
    preception_model_ckpt = 'LAVIS/LMDrive-vision-encoder-r50-v1.0/vision-encoder-r50.pth.tar'
    llm_model = 'LAVIS/llava-v1.5-7b'
    lad_drive_ckpt = '/beegfs/scratch/workspace/es_kafeit00-lang-traj/models/lad_drive_512/checkpoint_5.pth'

    abstract_lanes = False
    
    plan_anchor_path = "anchors/kmeans_anchors_lmdrive_20_modes_5_poses.npy"
    diffusion_hidden_dim = 512
    
    sample_rate = 2
    
    hd_visu = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
