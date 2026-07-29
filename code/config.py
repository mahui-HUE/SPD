import os
import numpy as np

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 

class Config_MBM_EEG(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 100
        self.clip_grad = 0.8
        
        # Model Parameters
        # self.mask_ratio = 0.1
        self.mask_ratio = 0.75
        self.patch_size = 4 #  1
        self.embed_dim = 1024 #256 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512 #128
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0
        #数据集设置
        self.root_path = '../DreamDiffusion_20250917_review/'
        self.output_path = '../DreamDiffusion_20250917_review/exps/'
        self.dataset_dir = r'/home/mahui/Dataset/EEG-ImageNet-Dataset'
        self.data_file = 'EEG-ImageNet_2.pth'
        self.subject = -1
        self.granularity = 'all'
        self.eeg_data_chan = 128
        self.eeg_data_len = 400
        self.eeg_normorlize=False# EEG-ImageNet按官方预处理，仅使用eeg_data[:, 40:440]
        self.eeg_scale = 10000.0  # EEG-ImageNet固定尺度放大，不做归一化
        self.eeg_preprocess = 'channel_minmax'
        self.eeg_zscore_eps = 1e-8
        self.things_sessions = None
        self.things_subjects = None
        self.things_tmin = -0.2
        self.things_tmax = 0.8
        self.things_epoch_crop = 'poststim'
        self.things_event_seed = 20200220
        self.things_test_max_rep = 1
        self.things_cache_sessions = 1
        # Project setting
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0


class Config_MC_EEG(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
        # --------------------------------------------
        # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 100
        self.clip_grad = 0.8

        # Model Parameters
        # self.mask_ratio = 0.1
        self.mask_ratio = 0.25#屏蔽通道的比例 0.125 0.25 0.5
        self.patch_size = 4  # 1
        self.embed_dim = 1024  # 256 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512  # 128
        self.depth = 24 #原24个vit块
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0
        self.position_flag = True
        self.fixed_electrode_indices = None
        self.fixed_electrode_index_base = 0
        self.position_ablation = 'none'
        self.position_ablation_seed = 0
        self.position_ablation_ref_index = 0
        self.position_ablation_virtual_coord = None
        self.spatial_depth = 3
        self.vit_order = 'spatial_temporal'
        # 数据集设置
        self.dataset_name = 'EEGCVPR'
        self.root_path = '../DreamDiffusion_20250917_review/'
        self.output_path = '../DreamDiffusion_20250917_review/exps/'
        self.dataset_dir = r'/home/mahui/Dataset/EEGCVPR40'
        self.data_file = 'EEG-ImageNet_2.pth'
        self.eeg_signals_path = '/home/mahui/Dataset/EEGCVPR40/eeg_5_95_std.pth'
        self.imagenet_path = '/home/mahui/Dataset/IMAGENET/train'
        self.splits_path = '/home/mahui/Dataset/EEGCVPR40/block_splits_by_image_all.pth'
        self.split_num = 0
        self.xyz_brain='/home/mahui/Dataset/EEGCVPR40/XYZ_Brain_EEGCVPR128.pth' #坐标文件
        self.subject = -1
        self.granularity = 'coarse'
        self.eeg_data_chan = 62
        self.eeg_data_len = 400
        self.eeg_normorlize = False  # EEG-ImageNet按官方预处理，仅使用eeg_data[:, 40:440]
        self.eeg_scale = 10000.0  # EEG-ImageNet固定尺度放大，不做归一化
        self.eeg_preprocess = 'channel_minmax'
        self.eeg_zscore_eps = 1e-8
        self.things_sessions = None
        self.things_subjects = None
        self.things_tmin = -0.2
        self.things_tmax = 0.8
        self.things_epoch_crop = 'poststim'
        self.things_event_seed = 20200220
        self.things_test_max_rep = 1
        self.things_cache_sessions = 1
        self.things_channel_set = 'all63'
        self.things_eeg_preprocess = 'session_channel_zscore'
        self.things_resample = 'polyphase'
        self.things_lowpass_hz = 0.0
        self.things_train_aggregation = 'image'
        self.things_train_repetition_mode = 'random'
        self.things_test_repetition_mode = 'mean'
        self.things_repetition_reduce_stage = 'processed'
        self.things_random_image_per_concept = False
        self.things_processed_cache_items = 64
        self.things_concept_clip_path = None
        self.things_freeze_eeg_encoder = False
        self.things_eeg_encoder_lr_scale = 1.0
        self.things_checkpoint_interval = 5
        # Project setting
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None  # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0

class Config_MC_EEG_finetune(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
        # --------------------------------------------
        # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 100
        self.clip_grad = 0.8

        # Model Parameters
        # self.mask_ratio = 0.1
        self.mask_ratio = 0.25#屏蔽通道的比例 0.125 0.25 0.5
        self.patch_size = 4  # 1
        self.embed_dim = 1024  # 256 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512  # 128
        self.depth = 24 #个vit块
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0
        self.position_flag = True
        self.fixed_electrode_indices = None
        self.fixed_electrode_index_base = 0
        self.position_ablation = 'none'
        self.position_ablation_seed = 0
        self.position_ablation_ref_index = 0
        self.position_ablation_virtual_coord = None
        self.spatial_depth = 3
        self.vit_order = 'spatial_temporal'
        # 数据集设置
        self.root_path = '../DreamDiffusion_20250917_review/'
        self.output_path = '../DreamDiffusion_20250917_review/exps/'
        self.dataset_dir = r'/home/mahui/Dataset/EEG-ImageNet-Dataset'
        self.data_file = 'EEG-ImageNet_2.pth'
        self.xyz_brain='/home/mahui/Dataset/EEG-ImageNet-Dataset/XYZ_Brain.pth' #坐标文件
        self.subject = -1
        self.granularity = 'all'
        self.targetChannels = 128
        self.targetSize = 400
        self.eeg_data_chan = self.targetChannels
        self.eeg_data_len = self.targetSize
        self.eeg_normorlize = False  # EEG-ImageNet按官方预处理，仅使用eeg_data[:, 40:440]
        self.eeg_scale = 10000.0  # EEG-ImageNet固定尺度放大，不做归一化
        self.eeg_preprocess = 'channel_minmax'
        self.eeg_zscore_eps = 1e-8
        self.things_sessions = None
        self.things_subjects = None
        self.things_tmin = -0.2
        self.things_tmax = 0.8
        self.things_event_seed = 20200220
        self.things_test_max_rep = 1
        self.things_cache_sessions = 1
        # Project setting
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None  # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0


class Config_EEG_finetune(Config_MBM_finetune):
    def __init__(self):
        
        # Project setting
        self.root_path = '../DreamDiffusion_20250917_review/'
        # self.root_path = '.'
        self.output_path = '../DreamDiffusion_20250917_review/exps/'

        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')

        self.dataset = 'EEG' 
        self.pretrain_mbm_path = '../DreamDiffusion_20250917_review/pretrains/eeg_pretrain/checkpoint.pth'

        self.include_nonavg_test = True


        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 16 if self.dataset == 'GOD' else 4 
        self.mask_ratio = 0.5
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0
        
class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '../DreamDiffusion_20250917_review/'
        self.output_path = '../DreamDiffusion_20250917_review/exps/'

        # self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4  # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0
        self.split_ratio = 0.975  # 数据集划分比率
        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')

        self.dataset = 'EEG'
        self.pretrain_mbm_path = None

        self.img_size = 128  # 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 500  # 虽然此处是500，但使用命令传入的是300,300与论文中一致

        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True  # False
        self.cls_tune = False
        # self.subject = 0
        self.eval_avg = True
        self.eval_pair_metrics = 'ssim'
        self.class_num_trials = 50

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None

        # 增加eeg数据集参数
        self.dataset_dir = '/home/mahui/Dataset/EEG-ImageNet-Dataset'
        self.data_file = 'EEG-ImageNet_1.pth'
        self.eeg_data_chan = 62
        self.eeg_data_len = 400
        self.eeg_normorlize = False  # 是否启用自定义eeg归一化
        self.eeg_scale = 10000.0  # EEG-ImageNet固定尺度放大，不做归一化
        self.eeg_preprocess = 'channel_minmax'
        self.eeg_zscore_eps = 1e-8
        self.subject = 0  # 两个文件共16个受试者，序号为0-15，选择第一个文件的第一个受试者。
        self.granularity = 'all'


class Config_Generative_Model_ours:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '../DreamDiffusion_20250917_review/'
        self.output_path = '../DreamDiffusion_20250917_review/exps/'

        # self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4  # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0
        self.split_ratio = 0.6 #数据集划分比率 0.975
        self.eeg_signals_path = '/home/mahui/Dataset/EEGCVPR40/eeg_5_95_std.pth'
        self.imagenet_path = '/home/mahui/Dataset/IMAGENET/train'
        self.splits_path = '/home/mahui/Dataset/EEGCVPR40/block_splits_by_image_single.pth'
        self.split_num = 0
        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')

        self.dataset = 'EEGCVPR'
        self.pretrain_mbm_path = None

        self.img_size = 256  # DreamDiffusion论文中512，我们在清华大学数据集设置的128

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 500  # 虽然此处是500，但使用命令传入的是300,300与论文中一致

        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True  # False
        self.cls_tune = False
        # self.subject = 0
        self.eval_avg = True
        self.eval_pair_metrics = 'ssim'
        self.class_num_trials = 50
        self.overfit_samples = 0
        self.overfit_seed = 2022
        self.overfit_eval_limit = 0

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None

        # 增加eeg数据集参数
        self.dataset_dir = '/home/mahui/Dataset/EEGCVPR40'
        self.data_file = 'EEG-ImageNet_1.pth'
        self.eeg_data_chan = 128
        self.eeg_data_len = 512
        self.n_way = 50 #49类负样本与1个正样本对比，不管数据集中有多少类图像，统一默认50
        self.mask_ratio = 0.75
        self.position_flag = True
        self.fixed_electrode_indices = None
        self.fixed_electrode_index_base = 0
        self.position_ablation = 'none'
        self.position_ablation_seed = 0
        self.position_ablation_ref_index = 0
        self.position_ablation_virtual_coord = None
        self.spatial_depth = 3
        self.vit_order = 'spatial_temporal'
        self.eeg_normorlize = False  # 是否启用自定义eeg归一化
        self.eeg_scale = 10000.0  # EEG-ImageNet分支使用，EEGCVPR分支不使用
        self.eeg_preprocess = 'channel_minmax'
        self.eeg_zscore_eps = 1e-8
        self.things_sessions = None
        self.things_subjects = None
        self.things_tmin = -0.2
        self.things_tmax = 0.8
        self.things_event_seed = 20200220
        self.things_test_max_rep = 1
        self.things_cache_sessions = 1
        self.things_channel_set = 'all63'
        self.things_epoch_crop = 'poststim'
        self.things_eeg_preprocess = 'session_channel_zscore'
        self.things_resample = 'polyphase'
        self.things_lowpass_hz = 0.0
        self.things_train_aggregation = 'image'
        self.things_train_repetition_mode = 'mean'
        self.things_test_repetition_mode = 'mean'
        self.things_repetition_reduce_stage = 'processed'
        self.things_random_image_per_concept = False
        self.things_processed_cache_items = 64
        self.things_concept_clip_path = None
        self.things_freeze_eeg_encoder = False
        self.things_eeg_encoder_lr_scale = 1.0
        self.things_checkpoint_interval = 5
        self.subject = 4  # 联合训练阶段固定使用被试者4。
        self.granularity = 'coarse'
        self.xyz_brain = '/home/mahui/Dataset/EEGCVPR40/XYZ_Brain_EEGCVPR128.pth' #坐标文件

class Config_Generative_Model_ours_ThoughtViz:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '../DreamDiffusion_20250917_review/'
        self.output_path = '../DreamDiffusion_20250917_review/exps/'

        # self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4  # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0
        #self.split_ratio = 0.975 #数据集划分比率
        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')

        self.dataset = 'EEG'
        self.pretrain_mbm_path = None

        self.img_size = 128  # 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 500  # 虽然此处是500，但使用命令传入的是300,300与论文中一致

        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True  # False
        self.cls_tune = False
        # self.subject = 0
        self.eval_avg = True
        self.eval_pair_metrics = 'ssim'
        self.class_num_trials = 50

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None

        # 增加eeg数据集参数
        self.dataset_dir = '/home/mahui/Dataset/EEG-ImageNet-Dataset'
        self.data_file = 'EEG-ImageNet_1.pth'
        self.eeg_data_chan = 128
        self.eeg_data_len = 400
        self.eeg_normorlize = False  # 是否启用自定义eeg归一化
        self.eeg_scale = 10000.0  # EEG-ImageNet固定尺度放大，不做归一化
        self.eeg_preprocess = 'channel_minmax'
        self.eeg_zscore_eps = 1e-8
        self.things_sessions = None
        self.things_subjects = None
        self.things_tmin = -0.2
        self.things_tmax = 0.8
        self.things_event_seed = 20200220
        self.things_test_max_rep = 1
        self.things_cache_sessions = 1
        self.subject = 4  # 联合训练阶段固定使用被试者4。
        self.granularity = 'all'
        self.xyz_brain = '/home/mahui/Dataset/EEGCVPR40/XYZ_Brain_EEGCVPR128.pth' #坐标文件



class Config_Cls_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '../DreamDiffusion_20250917_review/'
        self.output_path = '../DreamDiffusion_20250917_review/exps/'

        # self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_14_70_std.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')
        
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 50
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.15
        self.global_pool = False
        self.use_time_cond = False
        self.clip_tune = False
        self.subject = 4
        self.eval_avg = True
        self.eval_pair_metrics = 'ssim'
        self.class_num_trials = 50

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None
