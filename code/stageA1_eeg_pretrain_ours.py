import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt
import wandb
import copy

from dataset import eeg_pretrain_dataset_ours
from config import Config_MC_EEG,Config_MC_EEG_finetune
# from dataset import eeg_pretrain_dataset
from dataset import MyEEGDataset,Splitter,EEGCVPR_EEG,THINGSEEG2RawDataset,split_eeg_imagenet_random
from sc_mbm.mae_for_eeg import MChan_AEforEEG,MChan_AEforEEG_V2,MChan_AEforEEG_V3,MChan_AEforEEG_V4
from sc_mbm.trainer import train_one_epoch_ours
from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

class wandb_logger:
    def __init__(self, config):
        wandb.init(
                    project="DreamDiffusion_20250917_review",
                    anonymous="allow",
                    group='stageA_sc-mbm',
                    config=config,
                    reinit=True)

        self.config = config
        self.step = None
    
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

def get_args_parser():
    parser = argparse.ArgumentParser('MBM pre-training for fMRI', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)

    # Model Parameters
    parser.add_argument('--mask_ratio', type=float)
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--decoder_embed_dim', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--decoder_num_heads', type=int)
    parser.add_argument('--mlp_ratio', type=float)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--roi', type=str)
    parser.add_argument('--aug_times', type=int)
    parser.add_argument('--num_sub_limit', type=int)

    parser.add_argument('--include_hcp', type=bool)
    parser.add_argument('--include_kam', type=bool)

    parser.add_argument('--use_nature_img_loss', type=bool)
    parser.add_argument('--img_recon_weight', type=float)
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)

    # EEG-ImageNet数据集参数
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_dir', default=None, type=str,
                        help='path to EEG-ImageNet dataset')
    parser.add_argument('--data_file', default=None, type=str, help='file of EEG-ImageNet dataset')#使用第二个文件作预训练
    parser.add_argument('--eeg_signals_path', default=None, type=str)
    parser.add_argument('--imagenet_path', default=None, type=str)
    parser.add_argument('--splits_path', default=None, type=str)
    parser.add_argument('--split_num', default=0, type=int)
    parser.add_argument('--subject', default=-1, type=int, help='subject index')  # 选哪些人的EEG，-1默认全选
    parser.add_argument('--granularity', default='coarse', type=str, )
    parser.add_argument('--eeg_normorlize', action='store_true', default=None,
                        help='Deprecated for EEG-ImageNet; official preprocessing uses raw eeg_data[:, 40:440].')
    parser.add_argument('--eeg_scale', default=None, type=float,
                        help='Fixed scale for EEG-ImageNet raw eeg_data[:, 40:440].')
    parser.add_argument('--eeg_preprocess', default=None, type=str,
                        choices=['channel_minmax', 'scale', 'channel_zscore', 'none'],
                        help='EEG-ImageNet preprocessing mode.')
    parser.add_argument('--eeg_zscore_eps', default=None, type=float,
                        help='Epsilon for EEG-ImageNet channel-wise z-score.')
    parser.add_argument('--xyz_brain',default=None,type=str)
    parser.add_argument('--eeg_data_len', default=None, type=int)
    parser.add_argument('--eeg_data_chan', default=None, type=int)
    parser.add_argument('--position_flag', action='store_false',help='当命令行不指定此参数时，位置编码为启用状态')
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
    parser.add_argument('--things_subjects', default=None, type=str,
                        help='THINGS-EEG2 subjects for pretraining, e.g. "1,2,3" or "sub-01,sub-02". Overrides --subject.')
    parser.add_argument('--things_tmin', default=None, type=float,
                        help='THINGS-EEG2 epoch start in seconds relative to stimulus onset. Default follows official preprocessing: -0.2.')
    parser.add_argument('--things_tmax', default=None, type=float,
                        help='THINGS-EEG2 epoch end in seconds relative to stimulus onset. Default follows official preprocessing: 0.8.')
    parser.add_argument('--things_epoch_crop', default=None, type=str,
                        choices=['full', 'poststim'],
                        help='Keep the full baseline-inclusive epoch or only post-stimulus samples.')
    parser.add_argument('--things_event_seed', default=None, type=int,
                        help='Random seed for selecting THINGS-EEG2 repetitions when a condition has more than the official max repetitions.')
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
                        help='THINGS-EEG2 repetition handling during pretraining.')
    parser.add_argument('--things_test_repetition_mode', default=None, type=str,
                        choices=['random', 'mean', 'first'],
                        help='THINGS-EEG2 repetition handling during evaluation.')
    parser.add_argument('--things_repetition_reduce_stage', default=None, type=str,
                        choices=['raw', 'processed'],
                        help='Average repetitions before or after THINGS-EEG2 preprocessing.')
    parser.add_argument('--things_processed_cache_items', default=None, type=int,
                        help='Number of processed THINGS-EEG2 concept tensors cached per worker.')
    # parser.add_argument('--targetChannels',type=int,default=62,help='number of target eeg channels')
    # parser.add_argument('--targetSize',type=int,default=440,help='target eeg len')
                        
    return parser

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def main(config):
    # print('num of gpu:')
    # print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
    output_path = os.path.join(config.root_path, 'results', 'eeg_pretrain',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    # logger = wandb_logger(config) if config.local_rank == 0 else None
    logger = None
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # create dataset and dataloader
    # dataset_pretrain = eeg_pretrain_dataset(path='../DreamDiffusion_20250917_review/datasets/mne_data/', roi=config.roi, patch_size=config.patch_size,
    #             transform=fmri_transform, aug_times=config.aug_times, num_sub_limit=config.num_sub_limit,
    #             include_kam=config.include_kam, include_hcp=config.include_hcp)
    # if config.targetChannels == 62:#使用EEG-ImageNet数据集做为训练集
    #     dataset_pretrain = MyEEGDataset(config)
    # elif config.targetChannels == 128:#为"EEGCVPR40"或者"ThoughtViz"做预训练,但仍然使用EEG-ImageNet数据集做为训练集
    #     dataset_pretrain = eeg_pretrain_dataset_ours(target_data_len=config.targetSize,target_data_chan=config.targetChannels)
    # else:
    #     raise NotImplementedError
    if config.dataset_name == 'EEG-ImageNet':#使用EEG-ImageNet数据集做为训练集
        config.dataset_dir = '/home/mahui/Dataset/EEG-ImageNet-Dataset'
        full_dataset = MyEEGDataset(config)
        train_index, test_index = split_eeg_imagenet_random(
            full_dataset.data,
            train_ratio=0.6,
            seed=getattr(config, 'seed', 2022))
        print('EEG-ImageNet预训练整体随机划分：训练比例 0.600，测试比例 0.400')
        full_dataset.fit_eeg_zscore(train_index)
        dataset_pretrain = Subset(full_dataset, train_index)
        dataset_pretrain.data_len = full_dataset.data_len
        dataset_pretrain.data_chan = full_dataset.data_chan
        config.eeg_data_len = dataset_pretrain.data_len
        config.eeg_data_chan = dataset_pretrain.data_chan
        test_dataset = Subset(full_dataset, test_index)
        test_dataset.data_len = full_dataset.data_len
        test_dataset.data_chan = full_dataset.data_chan
        print('训练集大小:', len(dataset_pretrain))
        print('测试集大小:', len(test_dataset))
    # elif config.dataset_name == 'ThoughtViz':#
    #     dataset_pretrain = ThoughtViz(config)
    elif config.dataset_name == 'EEGCVPR':
        eeg_signals_path = getattr(config, 'eeg_signals_path', None) or '/home/mahui/Dataset/EEGCVPR40/eeg_5_95_std.pth'
        imagenet_path = getattr(config, 'imagenet_path', None) or '/home/mahui/Dataset/IMAGENET/train'
        dataset_pretrain = EEGCVPR_EEG(config, eeg_signals_path=eeg_signals_path, imagenet_path=imagenet_path, exclude_subject=4)
        print('预训练数据与original一致：使用EEGCVPR_EEG并排除subject 4')
    elif config.dataset_name in ['THINGS-EEG2', 'THINGS_EEG2', 'THINGSEEG2']:
        dataset_pretrain = THINGSEEG2RawDataset(config, split='train', include_images=False)
        config.eeg_data_len = dataset_pretrain.data_len
        config.eeg_data_chan = dataset_pretrain.data_chan
        print('THINGS-EEG2预训练：读取raw continuous EEG并按配置切分epoch')
    else:
        raise NotImplementedError

    print(f'Dataset size: {len(dataset_pretrain)}\n Time len: {dataset_pretrain.data_len}')
    sampler = torch.utils.data.DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

    dataloader_eeg = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, 
                shuffle=(sampler is None), pin_memory=True)

    # create model
    config.time_len=dataset_pretrain.data_len
    # model = MAEforEEG(time_len=dataset_pretrain.data_len, patch_size=config.patch_size, embed_dim=config.embed_dim,in_chans=dataset_pretrain.data_chan,
    #                 decoder_embed_dim=config.decoder_embed_dim, depth=config.depth,
    #                 num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
    #                 focus_range=config.focus_range, focus_rate=config.focus_rate,
    #                 img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss)
    # model = MChan_AEforEEG(time_len=dataset_pretrain.data_len, patch_size=config.patch_size,embed_dim=config.embed_dim,in_chans=dataset_pretrain.data_chan,mask_ratio=config.mask_ratio,
    #                        decoder_embed_dim=config.decoder_embed_dim,depth=config.depth,num_heads=config.num_heads,decoder_num_heads=config.decoder_num_heads,mlp_ratio=config.mlp_ratio,
    #                        xyz_brain=config.xyz_brain)
    # model = MChan_AEforEEG_V2(time_len=dataset_pretrain.data_len, patch_size=config.patch_size, embed_dim=config.embed_dim,
    #                        in_chans=dataset_pretrain.data_chan, mask_ratio=config.mask_ratio,
    #                        decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, num_heads=config.num_heads,
    #                        decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
    #                        xyz_brain=config.xyz_brain)
    # model = MChan_AEforEEG_V3(time_len=dataset_pretrain.data_len, patch_size=config.patch_size,
    #                           embed_dim=config.embed_dim,
    #                           in_chans=dataset_pretrain.data_chan, mask_ratio=config.mask_ratio,
    #                           decoder_embed_dim=config.decoder_embed_dim, depth=config.depth,
    #                           num_heads=config.num_heads,
    #                           decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,position_flag=True,
    #                           xyz_brain=config.xyz_brain)
    model = MChan_AEforEEG_V4(time_len=dataset_pretrain.data_len, patch_size=config.patch_size, embed_dim=config.embed_dim,
                           in_chans=dataset_pretrain.data_chan, mask_ratio=config.mask_ratio,
                           decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, num_heads=config.num_heads,
                           decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
                           xyz_brain=config.xyz_brain,position_flag=config.position_flag,
                           fixed_electrode_indices=config.fixed_electrode_indices,
                           fixed_electrode_index_base=config.fixed_electrode_index_base,
                           position_ablation=getattr(config, 'position_ablation', 'none'),
                           position_ablation_seed=getattr(config, 'position_ablation_seed', 0),
                           position_ablation_ref_index=getattr(config, 'position_ablation_ref_index', 0),
                           position_ablation_virtual_coord=getattr(config, 'position_ablation_virtual_coord', None),
                           spatial_depth=getattr(config, 'spatial_depth', None),
                           vit_order=getattr(config, 'vit_order', 'spatial_temporal'))
    model.to(device)
    model_without_ddp = model
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=config.use_nature_img_loss)

    param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)


    start_time = time.time()
    print('Start Training the EEG MAE ... ...')
    img_feature_extractor = None
    preprocess = None
    if config.use_nature_img_loss:
        from torchvision.models import resnet50, ResNet50_Weights
        from torchvision.models.feature_extraction import create_feature_extractor
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        m = resnet50(weights=weights)   
        img_feature_extractor = create_feature_extractor(m, return_nodes={f'layer2': 'layer2'}).to(device).eval()
        for param in img_feature_extractor.parameters():
            param.requires_grad = False

    for ep in range(config.num_epoch):
        
        if torch.cuda.device_count() > 1: 
            sampler.set_epoch(ep) # to shuffle the data at every epoch
        return_value = train_one_epoch_ours(model, dataloader_eeg, optimizer, device, ep, loss_scaler, logger, config, start_time, model_without_ddp,
                            img_feature_extractor, preprocess)

        if (ep % 20 == 0 or ep + 1 == config.num_epoch) and config.local_rank == 0: #and ep != 0
            # save models
        # if True:
            save_model(config, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path,'checkpoints'))
            # plot figures
            plot_recon_figures3(model, device, dataset_pretrain, output_path, 5, config, logger, model_without_ddp)
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:
        logger.finish()
    return

@torch.no_grad()
def plot_recon_figures(model, device, dataset, output_path, num_figures = 5, config=None, logger=None, model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Masked Ground-truth')
    axs[0,2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['eeg']
        sample = sample.to(device)
        _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
        # sample_with_mask = model_without_ddp.patchify(sample.transpose(1,2))[0].to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_without_ddp.patch_size)
        # pred = model_without_ddp.unpatchify(pred.transpose(1,2)).to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        # sample = sample.to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        pred = model_without_ddp.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
        # pred = model_without_ddp.unpatchify(model_without_ddp.patchify(sample.transpose(1,2))).to('cpu').squeeze(0)[0].numpy()
        sample = sample.to('cpu').squeeze(0)[0].numpy()
        mask = mask.to('cpu').numpy().reshape(-1)

        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask,mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)


@torch.no_grad()
def plot_recon_figures2(model, device, dataset, output_path, num_figures = 5, config=None, logger=None, model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 2, figsize=(20,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    # axs[0,1].set_title('Masked Ground-truth')
    axs[0,1].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['eeg']
        sample = sample.to(device)
        _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
        # sample_with_mask = model_without_ddp.patchify(sample.transpose(1,2))[0].to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_without_ddp.patch_size)
        # pred = model_without_ddp.unpatchify(pred.transpose(1,2)).to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        # sample = sample.to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        pred = model_without_ddp.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
        # pred = model_without_ddp.unpatchify(model_without_ddp.patchify(sample.transpose(1,2))).to('cpu').squeeze(0)[0].numpy()
        sample = sample.to('cpu').squeeze(0)[0].numpy()
        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)

        ax[1].plot(x_axis, pred)
        ax[1].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[1].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)

@torch.no_grad()
def plot_recon_figures3(model, device, dataset, output_path, num_figures = 5, config=None, logger=None, model_without_ddp=None):
    #5是样本数 8是每个样本丢掉的通道数
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 2, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    # axs[0,1].set_title('Masked Ground-truth')
    axs[0,1].set_title('Reconstruction')
    #sample(N,channels,timepoints)
    for ax in axs:
        sample = next(iter(dataloader))['eeg']
        sample = sample.to(device)#(N,channels,timepoints)
        #pred(B,T,C)
        _,pred, mask = model(sample)
        # sample_with_mask = model_without_ddp.patchify(sample.transpose(1,2))[0].to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_without_ddp.patch_size)
        # pred = model_without_ddp.unpatchify(pred.transpose(1,2)).to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        # sample = sample.to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        # print('*'*50)
        # print('pred.shape1', pred.shape)
        # pred = model_without_ddp.unpatchify(pred).to('cpu').squeeze(0).numpy()#pred(B=1,timepoints,channels)--(timepoints,channels)--(channels,timepoints)
        pred = pred.to('cpu').squeeze(
            0).numpy()  # pred(B=1,channels,timepoints)(channels,timepoints)
        # print('*' * 50)
        # print('pred.shape2',pred.shape)
        mask = mask.to('cpu').squeeze(0).numpy()#mask(B=1,channels)--(channels)
        # pred = model_without_ddp.unpatchify(model_without_ddp.patchify(sample.transpose(1,2))).to('cpu').squeeze(0)[0].numpy()

        sample = sample.to('cpu').squeeze(0).numpy()#sample(B=1,channels,timepoints)--(channels,timepoints)
        x_axis = np.arange(0, sample.shape[-1])
        channel_index = 0
        for s,p,m in zip(sample, pred, mask):#sample(channels,timepoints) pred(channels,timepoints) mask(channels)
            if m == 1:#画出被屏蔽的部分对比
                ax[0].plot(x_axis, s)
                ax[1].plot(x_axis, p)
                cor = np.corrcoef([p, s])[0,1]

                ax[0].set_ylabel('channel_index: %d' % channel_index,weight='bold')
                ax[1].set_ylabel('cor: %.4f' % cor, weight='bold')
                ax[0].yaxis.set_label_position("left")
                ax[1].yaxis.set_label_position("right")
                break#对于每个样本来说，屏蔽的通道是随机的，因此，每次选的第一个被屏蔽的通道也是随机的
            channel_index = channel_index + 1
    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)


def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_MC_EEG()
    # config = Config_MC_EEG_finetune()
    config = update_config(args, config)
    main(config)
