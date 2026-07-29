import os, sys
import numpy as np
import torch
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
import copy

from config import Config_Generative_Model_ours
# own code
from config import Config_Generative_Model
# from dataset import  create_EEG_dataset
from dataset import create_myEEGImageNetDataset
from dc_ldm.ldm_for_eeg_ours import eLDM
from eval_metrics import get_similarity_metric
from scipy.stats import combine_pvalues

#设置此项主要是为了防止打开太多文件的错误
torch.multiprocessing.set_sharing_strategy('file_system')

def wandb_init(config, output_path):
    # wandb.init( project='DreamDiffusion_20250917_review',
    #             group="stageB_dc-ldm",
    #             anonymous="allow",
    #             config=config,
    #             reinit=True)
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def parse_eval_pair_metrics(pair_metrics):
    if pair_metrics is None:
        return ['ssim']
    if isinstance(pair_metrics, str):
        pair_metrics = [m.strip() for m in pair_metrics.split(',') if m.strip()]
    return list(pair_metrics)


def get_eval_metric(samples, avg=True, pair_metrics='ssim', class_num_trials=50):
    metric_list = parse_eval_pair_metrics(pair_metrics)
    res_list = []

    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))
    res_part = []
    std_part = []
    p_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None,
                                    n_way=50, num_trials=class_num_trials, top_k=1, device='cuda',
                                    return_stats=True)
        res_part.append(np.mean(res['acc']))
        std_part.append(res['overall_std'])
        p_part.append(res['combined_p'])
    max_idx = int(np.argmax(res_part))
    valid_p = [p for p in p_part if not np.isnan(p)]
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    res_list.append(np.mean(std_part))
    res_list.append(std_part[max_idx])
    res_list.append(combine_pvalues(valid_p, method='fisher').pvalue if len(valid_p) > 0 else np.nan)
    res_list.append(p_part[max_idx])
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    metric_list.append('top-1-class std')
    metric_list.append('top-1-class (max) std')
    metric_list.append('top-1-class p')
    metric_list.append('top-1-class (max) p')
    return res_list, metric_list
               
def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    grid, _ = generative_model.generate(eeg_latents_dataset_train, config.num_samples, 
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    # wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})

    if config.dataset in ['THINGS-EEG2', 'THINGS_EEG2', 'THINGSEEG2']:
        eval_limit = getattr(config, 'things_eval_limit', 0)
    else:
        eval_limit = getattr(config, 'overfit_eval_limit', 0)
    eval_limit = eval_limit if eval_limit is not None and eval_limit > 0 else None
    grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples, 
                config.ddim_steps, config.HW, eval_limit)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_test.png'))
    for sp_idx, imgs in enumerate(samples):
        for copy_idx, img in enumerate(imgs):
            img = rearrange(img, 'c h w -> h w c')
            Image.fromarray(img).save(os.path.join(config.output_path, 
                            f'./test{sp_idx}-{copy_idx}.png'))

    # wandb.log({f'summary/samples_test': wandb.Image(grid_imgs)})

    metric, metric_list = get_eval_metric(samples, avg=config.eval_avg,
                                          pair_metrics=getattr(config, 'eval_pair_metrics', 'ssim'),
                                          class_num_trials=getattr(config, 'class_num_trials', 50))
    pair_wise_metrics = {'mse', 'pcc', 'ssim', 'psm'}
    metric_dict = {}
    for k, v in zip(metric_list, metric):
        prefix = 'summary/pair-wise_' if k in pair_wise_metrics else 'summary/'
        metric_dict[f'{prefix}{k}'] = v
    print('评估结果为如下：')
    print(metric_dict)
    # wandb.log(metric_dict)

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def main(config):
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,

        # transforms.Resize((512, 512)),
        transforms.Resize((config.img_size, config.img_size)),
        random_crop(config.img_size-crop_pix, p=0.5),

        # transforms.Resize((512, 512)),
        transforms.Resize((config.img_size, config.img_size)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, 

        # transforms.Resize((512, 512)),
        transforms.Resize((config.img_size, config.img_size)),
        channel_last
    ])
    is_things_eeg2 = config.dataset in ['THINGS-EEG2', 'THINGS_EEG2', 'THINGSEEG2']
    if is_things_eeg2 or getattr(config, 'overfit_samples', 0) > 0:
        # THINGS-EEG2 uses a deterministic reconstruction target in formal training too.
        img_transform_train = img_transform_test
    # if config.dataset == 'EEG':
    if config.dataset is not None:

        # eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset(eeg_signals_path = config.eeg_signals_path, splits_path = config.splits_path,
        #         image_transform=[img_transform_train, img_transform_test], subject = config.subject)
        eeg_latents_dataset_train, eeg_latents_dataset_test = create_myEEGImageNetDataset(args=config,dataset_name=config.dataset,split_ratio=config.split_ratio,
                                                                                          image_transform=[
                                                                                              img_transform_train,
                                                                                              img_transform_test])
        # eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset_viz( image_transform=[img_transform_train, img_transform_test])
        # num_voxels = eeg_latents_dataset_train.data_len
        num_voxels = getattr(eeg_latents_dataset_train, 'data_len', config.eeg_data_len)
        config.eeg_data_len = num_voxels
        if hasattr(eeg_latents_dataset_train, 'data_chan'):
            config.eeg_data_chan = eeg_latents_dataset_train.data_chan

        if (is_things_eeg2
                and getattr(config, 'things_train_limit', 0) > 0
                and getattr(config, 'overfit_samples', 0) <= 0):
            train_limit = min(
                int(config.things_train_limit), len(eeg_latents_dataset_train))
            train_limit_seed = int(getattr(
                config, 'things_train_limit_seed', config.seed))
            generator = torch.Generator().manual_seed(train_limit_seed)
            train_indices = torch.randperm(
                len(eeg_latents_dataset_train), generator=generator
            )[:train_limit].tolist()
            eeg_latents_dataset_train = torch.utils.data.Subset(
                eeg_latents_dataset_train, train_indices)
            np.savetxt(
                os.path.join(config.output_path, 'things_train_indices.txt'),
                np.asarray(train_indices, dtype=np.int64), fmt='%d')
            print(f'THINGS-EEG2 pilot training subset: {train_limit} samples '
                  f'(seed={train_limit_seed}); official test set is unchanged.')

        if getattr(config, 'overfit_samples', 0) > 0:
            overfit_samples = min(int(config.overfit_samples), len(eeg_latents_dataset_train))
            if overfit_samples <= 0:
                raise ValueError('--overfit_samples must be greater than 0')
            overfit_seed = int(getattr(config, 'overfit_seed', config.seed))
            generator = torch.Generator().manual_seed(overfit_seed)
            overfit_indices = torch.randperm(
                len(eeg_latents_dataset_train), generator=generator
            )[:overfit_samples].tolist()
            source_train_dataset = eeg_latents_dataset_train
            eeg_latents_dataset_train = torch.utils.data.Subset(
                source_train_dataset, overfit_indices)
            eeg_latents_dataset_test = torch.utils.data.Subset(
                source_train_dataset, overfit_indices)
            np.savetxt(
                os.path.join(config.output_path, 'overfit_indices.txt'),
                np.asarray(overfit_indices, dtype=np.int64), fmt='%d')
            print(f'Overfit sanity check: train/validation share {overfit_samples} fixed samples '
                  f'(seed={overfit_seed}).')
            print(f'Overfit sample indices saved to: '
                  f'{os.path.join(config.output_path, "overfit_indices.txt")}')

    else:
        raise NotImplementedError
    # print(num_voxels)

    # prepare pretrained mbm 
    if config.pretrain_mbm_path is not None:
        pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')
    else:
        pretrain_mbm_metafile = None

    # create generateive model
    generative_model = eLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger, 
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond, clip_tune = config.clip_tune, cls_tune = config.cls_tune,
                runtime_config=config)
    
    # resume training if applicable
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        if (is_things_eeg2
                and getattr(config, 'things_clip_context_weight', 0.0) != 0):
            missing, unexpected = generative_model.model.load_state_dict(
                model_meta['model_state_dict'], strict=False)
            allowed_missing = {
                'cond_stage_model.things_clip_context_norm.weight',
                'cond_stage_model.things_clip_context_norm.bias',
                'model_ema.cond_stage_modelthings_clip_context_normweight',
                'model_ema.cond_stage_modelthings_clip_context_normbias',
            }
            disallowed_missing = set(missing) - allowed_missing
            if disallowed_missing or unexpected:
                raise RuntimeError(
                    f'Unexpected THINGS-EEG2 resume mismatch: '
                    f'missing={sorted(disallowed_missing)}, unexpected={unexpected}')
            print('THINGS-EEG2 context injection initialized while resuming:',
                  sorted(set(missing) & allowed_missing))
        else:
            generative_model.model.load_state_dict(model_meta['model_state_dict'])
        print('model resumed')
    # finetune the model
    trainer = create_trainer(
        config.num_epoch,
        config.precision,
        config.accumulate_grad,
        config.logger,
        check_val_every_n_epoch=2,
        disable_validation=is_things_eeg2)
    generative_model.finetune(trainer, eeg_latents_dataset_train, eeg_latents_dataset_test,
                config.batch_size, config.lr, config.output_path, config=config)

    # generate images
    # generate limited train images and generate images for subjects seperately
    generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config)
    print('输入输出图像分辨率为：', config.img_size, '预测的对象subject为：', config.subject)
    print('全部程序执行完毕')
    return

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    # parser.add_argument('--root_path', type=str, default = '../DreamDiffusion_20250917_review/')
    parser.add_argument('--root_path', type=str, default='../DreamDiffusion_20250917_review/')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Optional output root; a timestamped results/generation folder is created below it.')
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--depth', type=int)

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)
    parser.add_argument('--clip_tune', action='store_true', default=None,
                        help='Enable CLIP alignment loss during joint training.')
    parser.add_argument('--no_clip_tune', action='store_false', dest='clip_tune',
                        help='Disable CLIP alignment loss during joint training.')

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=bool)
    parser.add_argument('--eval_avg', type=bool)
    parser.add_argument('--eval_pair_metrics', default=None, type=str,
                        help='Comma-separated pair-wise metrics. Default config keeps only ssim.')
    parser.add_argument('--class_num_trials', default=None, type=int,
                        help='Number of trials for top-1 class evaluation.')
    # EEG-ImageNet数据集参数
    parser.add_argument('--dataset_dir', default=None, type=str,
                        help='path to EEG-ImageNet dataset')
    parser.add_argument('--data_file', default=None, type=str, help='file of EEG-ImageNet dataset')
    parser.add_argument('--subject', default=None,type=int, help='subject index')  # 选哪些人的EEG，-1默认全选
    parser.add_argument('--granularity', default=None, type=str, )
    parser.add_argument('--eeg_data_chan',default=None,type=int,help='EEG-ImageNet channel')
    parser.add_argument('--eeg_data_len',default=None,type=int)
    parser.add_argument('--n_way',default=None,type=int,help='number class of images')#数据集有多少类图像
    parser.add_argument('--eeg_normorlize', action='store_true', default=None,
                        help='Deprecated for EEG-ImageNet; official preprocessing uses raw eeg_data[:, 40:440].')
    parser.add_argument('--eeg_scale', default=None, type=float,
                        help='Fixed scale for EEG-ImageNet raw eeg_data[:, 40:440].')
    parser.add_argument('--eeg_preprocess', default=None, type=str,
                        choices=['channel_minmax', 'scale', 'channel_zscore', 'none'],
                        help='EEG-ImageNet preprocessing mode.')
    parser.add_argument('--eeg_zscore_eps', default=None, type=float,
                        help='Epsilon for EEG-ImageNet channel-wise z-score.')
    parser.add_argument('--xyz_brain', default=None, type=str)
    parser.add_argument('--eeg_signals_path', default=None, type=str)
    parser.add_argument('--imagenet_path', default=None, type=str)
    parser.add_argument('--splits_path', default=None, type=str)
    parser.add_argument('--split_num', default=None, type=int)
    parser.add_argument('--mask_ratio', type=float)
    parser.add_argument('--position_flag', action='store_false', help='当命令行不指定此参数时，位置编码为启用状态')
    parser.add_argument('--fixed_electrode_indices', default=None, type=str,
                        help='Optional fixed electrode index file or comma-separated list. Leave unset for random masking.')
    parser.add_argument('--fixed_electrode_index_base', default=0, type=int,
                        help='Use 0 for zero-based indices and 1 for one-based indices.')
    parser.add_argument('--position_ablation', default=None, type=str,
                        choices=['none', 'shuffle_xyz', 'shuffle_region', 'shuffle_both', 'same_coord', 'virtual_coord'])
    parser.add_argument('--position_ablation_seed', default=None, type=int)
    parser.add_argument('--position_ablation_ref_index', default=None, type=int)
    parser.add_argument('--position_ablation_virtual_coord', default=None, type=str)
    parser.add_argument('--spatial_depth', default=None, type=int,
                        help='Number of spatial/channel ViT blocks. Temporal blocks are depth - spatial_depth.')
    parser.add_argument('--vit_order', default=None, type=str,
                        choices=['spatial_temporal', 'temporal_spatial'],
                        help='Order of spatial/channel and temporal ViT blocks.')
    parser.add_argument('--things_sessions', default=None, type=str,
                        help='THINGS-EEG2 sessions, e.g. "1,2" or "ses-01,ses-02". Default uses all sessions.')
    parser.add_argument('--things_tmin', default=None, type=float,
                        help='THINGS-EEG2 epoch start in seconds relative to stimulus onset. Default follows official preprocessing: -0.2.')
    parser.add_argument('--things_tmax', default=None, type=float,
                        help='THINGS-EEG2 epoch end in seconds relative to stimulus onset. Default follows official preprocessing: 0.8.')
    parser.add_argument('--things_epoch_crop', default=None, type=str,
                        choices=['full', 'poststim'],
                        help='Keep the full baseline-inclusive epoch or only post-stimulus samples.')
    parser.add_argument('--things_event_seed', default=None, type=int,
                        help='Random seed for selecting THINGS-EEG2 repetitions when a condition has more than the official max repetitions.')
    parser.add_argument('--things_test_max_rep', default=None, type=int,
                        help='Max THINGS-EEG2 test repetitions per image condition after merging sessions. Default is 1.')
    parser.add_argument('--things_cache_sessions', default=None, type=int,
                        help='Number of THINGS-EEG2 raw session arrays cached per worker.')
    parser.add_argument('--things_channel_set', default=None, type=str,
                        choices=['all63', 'official17'],
                        help='THINGS-EEG2 EEG channel subset. official17 matches the official preprocessed release.')
    parser.add_argument('--things_eeg_preprocess', default=None, type=str,
                        choices=['legacy_channel_minmax', 'global_minmax', 'global_zscore',
                                 'channel_zscore_sample', 'session_channel_zscore', 'none'],
                        help='THINGS-EEG2-only EEG normalization; legacy preserves old checkpoints.')
    parser.add_argument('--things_resample', default=None, type=str,
                        choices=['linear', 'polyphase'],
                        help='THINGS-EEG2-only temporal resampling method.')
    parser.add_argument('--things_lowpass_hz', default=None, type=float,
                        help='Optional THINGS-EEG2 low-pass cutoff in Hz; 0 disables filtering.')
    parser.add_argument('--things_train_aggregation', default=None, type=str,
                        choices=['image', 'concept'],
                        help='THINGS-EEG2 training EEG aggregation level.')
    parser.add_argument('--things_train_repetition_mode', default=None, type=str,
                        choices=['random', 'mean', 'first'],
                        help='THINGS-EEG2 repetition handling during joint training.')
    parser.add_argument('--things_test_repetition_mode', default=None, type=str,
                        choices=['random', 'mean', 'first'],
                        help='THINGS-EEG2 repetition handling during evaluation.')
    parser.add_argument('--things_repetition_reduce_stage', default=None, type=str,
                        choices=['raw', 'processed'],
                        help='Average repetitions before or after THINGS-EEG2 preprocessing.')
    parser.add_argument('--things_random_image_per_concept', action='store_true', default=None,
                        help='Use one concept item per epoch and randomly choose one of its images.')
    parser.add_argument('--things_processed_cache_items', default=None, type=int,
                        help='Number of processed THINGS-EEG2 concept tensors cached per worker.')
    parser.add_argument('--things_concept_clip_path', default=None, type=str,
                        help='Optional THINGS-EEG2 concept-level CLIP target cache.')
    parser.add_argument('--things_freeze_eeg_encoder', action='store_true', default=None,
                        help='Freeze the pretrained EEG backbone during THINGS-EEG2 joint training.')
    parser.add_argument('--no_things_freeze_eeg_encoder', action='store_false',
                        dest='things_freeze_eeg_encoder',
                        help='Allow THINGS-EEG2 joint training to update the EEG backbone.')
    parser.add_argument('--things_eeg_encoder_lr_scale', default=None, type=float,
                        help='THINGS-EEG2 EEG-backbone LR multiplier; 1 preserves the original optimizer.')
    parser.add_argument('--things_checkpoint_interval', default=None, type=int,
                        help='Save the THINGS-EEG2 latest checkpoint every N epochs.')
    parser.add_argument('--things_clip_loss', default=None, type=str,
                        choices=['cosine', 'cosine_contrastive'],
                        help='THINGS-EEG2-only CLIP alignment objective.')
    parser.add_argument('--things_clip_temperature', default=None, type=float,
                        help='Temperature for the THINGS-EEG2 contrastive CLIP loss.')
    parser.add_argument('--things_clip_cosine_weight', default=None, type=float,
                        help='Cosine component weight for THINGS-EEG2 CLIP alignment.')
    parser.add_argument('--things_clip_contrastive_weight', default=None, type=float,
                        help='Contrastive component weight for THINGS-EEG2 CLIP alignment.')
    parser.add_argument('--things_clip_queue_size', default=None, type=int,
                        help='Number of detached image CLIP embeddings used as THINGS-EEG2 negatives.')
    parser.add_argument('--things_clip_context_weight', default=None, type=float,
                        help='THINGS-EEG2-only residual weight for injecting aligned global CLIP context.')
    parser.add_argument('--things_train_limit', default=None, type=int,
                        help='THINGS-EEG2-only deterministic training subset size; 0 uses all conditions.')
    parser.add_argument('--things_train_limit_seed', default=None, type=int,
                        help='Seed for the THINGS-EEG2 pilot training subset.')
    parser.add_argument('--things_eval_limit', default=None, type=int,
                        help='THINGS-EEG2-only final generation/evaluation limit; 0 uses the full test set.')
    parser.add_argument('--overfit_samples', default=None, type=int,
                        help='Use a fixed subset of the training set for both training and validation.')
    parser.add_argument('--overfit_seed', default=None, type=int,
                        help='Random seed used to choose the fixed overfit subset.')
    parser.add_argument('--overfit_eval_limit', default=None, type=int,
                        help='Maximum number of overfit-subset samples generated for final evaluation.')

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


def update_things_config(args, config):
    if config.dataset not in ['THINGS-EEG2', 'THINGS_EEG2', 'THINGSEEG2']:
        return config
    defaults = {
        'things_clip_loss': 'cosine',
        'things_clip_temperature': 0.07,
        'things_clip_cosine_weight': 1.0,
        'things_clip_contrastive_weight': 0.1,
        'things_clip_queue_size': 0,
        'things_clip_context_weight': 0.0,
        'things_train_limit': 0,
        'things_train_limit_seed': config.seed,
        'things_eval_limit': 0,
        'things_channel_set': 'all63',
        'things_eeg_preprocess': 'legacy_channel_minmax',
        'things_epoch_crop': 'poststim',
        'things_resample': 'linear',
        'things_lowpass_hz': 0.0,
        'things_train_aggregation': 'image',
        'things_train_repetition_mode': 'mean',
        'things_test_repetition_mode': 'mean',
        'things_repetition_reduce_stage': 'processed',
        'things_random_image_per_concept': False,
        'things_processed_cache_items': 64,
        'things_concept_clip_path': None,
        'things_freeze_eeg_encoder': False,
        'things_eeg_encoder_lr_scale': 1.0,
        'things_checkpoint_interval': 5,
    }
    for attr, default in defaults.items():
        value = getattr(args, attr, None)
        if value is None:
            value = getattr(config, attr, default)
        setattr(config, attr, value)
    return config

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


class ThingsLatestCheckpoint(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        config = getattr(pl_module, 'main_config', None)
        if config is None or getattr(config, 'dataset', None) not in [
                'THINGS-EEG2', 'THINGS_EEG2', 'THINGSEEG2']:
            return
        output_path = getattr(pl_module, 'output_path', None)
        if output_path is None:
            return
        interval = max(1, int(getattr(config, 'things_checkpoint_interval', 5)))
        completed_epoch = trainer.current_epoch + 1
        if completed_epoch >= trainer.max_epochs or completed_epoch % interval != 0:
            return
        checkpoint_path = os.path.join(output_path, 'checkpoint_latest.pth')
        temporary_path = checkpoint_path + '.tmp'
        torch.save({
            'model_state_dict': pl_module.state_dict(),
            'config': config,
            'state': torch.random.get_rng_state(),
            'completed_epoch': trainer.current_epoch,
        }, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        print(f'THINGS-EEG2 latest checkpoint saved after epoch '
              f'{trainer.current_epoch}: {checkpoint_path}')


def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2, logger=None,
                   check_val_every_n_epoch=0, disable_validation=False):
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer_kwargs = dict(
        accelerator=acc,
        max_epochs=num_epoch,
        logger=logger,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_checkpointing=False,
        enable_model_summary=False,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
    if disable_validation:
        trainer_kwargs.update(
            num_sanity_val_steps=0,
            limit_val_batches=0,
            callbacks=[ThingsLatestCheckpoint()],
        )
        print('THINGS-EEG2 intermediate generative validation disabled; final evaluation is retained.')
    return pl.Trainer(**trainer_kwargs)
  
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model_ours()
    config = update_config(args, config)
    config = update_things_config(args, config)
    runtime_output_path = config.output_path

    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        ckp = config.checkpoint_path
        config = model_meta['config']
        config.checkpoint_path = ckp
        if config.dataset in ['THINGS-EEG2', 'THINGS_EEG2', 'THINGSEEG2']:
            config = update_config(args, config)
            config = update_things_config(args, config)
            config.output_path = runtime_output_path
        print('Resuming from checkpoint: {}'.format(config.checkpoint_path))

    if (config.dataset in ['THINGS-EEG2', 'THINGS_EEG2', 'THINGSEEG2']
            and args.ddim_steps is None):
        config.ddim_steps = 50

    output_path = os.path.join(config.output_path, 'results', 'generation',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    
    wandb_init(config, output_path)

    logger = TensorBoardLogger("logs", name="my_experiment")
    config.logger = logger
    # config.logger = None # logger
    main(config)
    wandb_finish()
