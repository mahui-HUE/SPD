import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_eeg import eeg_encoder_ours, eeg_encoder_ours_V2,eeg_encoder_ours_V3,eeg_encoder_ours_V4,classify_network, mapping
from PIL import Image
def create_model_from_config(config, num_voxels):
    # model = eeg_encoder_ours(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,in_chans=config.eeg_data_chan,mask_ratio=config.mask_ratio,
    #             depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,position_flag=True,xyz_brain=config.xyz_brain)
    # model = eeg_encoder_ours_V2(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
    #                          in_chans=config.eeg_data_chan, mask_ratio=config.mask_ratio,
    #                          depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
    #                          position_flag=True, xyz_brain=config.xyz_brain)
    # model = eeg_encoder_ours_V3(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
    #                          in_chans=config.eeg_data_chan,mask_ratio=0,
    #                          depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
    #                          position_flag=True, xyz_brain=config.xyz_brain)
    model = eeg_encoder_ours_V4(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                             in_chans=config.eeg_data_chan, mask_ratio=config.mask_ratio,
                             depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                             position_flag=getattr(config, 'position_flag', True), xyz_brain=config.xyz_brain,
                             fixed_electrode_indices=getattr(config, 'fixed_electrode_indices', None),
                             fixed_electrode_index_base=getattr(config, 'fixed_electrode_index_base', 0),
                             position_ablation=getattr(config, 'position_ablation', 'none'),
                             position_ablation_seed=getattr(config, 'position_ablation_seed', 0),
                             position_ablation_ref_index=getattr(config, 'position_ablation_ref_index', 0),
                             position_ablation_virtual_coord=getattr(config, 'position_ablation_virtual_coord', None),
                             spatial_depth=getattr(config, 'spatial_depth', None),
                             vit_order=getattr(config, 'vit_order', 'spatial_temporal'))
    return model

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels=400, cond_dim=1280, global_pool=True, clip_tune = True, cls_tune = False,
                 runtime_config=None):
        super().__init__()
        # prepare pretrained fmri mae 
        if metafile is not None:
            model_config = metafile['config']
            if runtime_config is not None and getattr(runtime_config, 'fixed_electrode_indices', None) is not None:
                model_config.fixed_electrode_indices = runtime_config.fixed_electrode_indices
                model_config.fixed_electrode_index_base = getattr(runtime_config, 'fixed_electrode_index_base', 0)
                print('使用运行时固定电极索引覆盖预训练配置:', model_config.fixed_electrode_indices)
            if runtime_config is not None:
                model_config.position_ablation = getattr(runtime_config, 'position_ablation', getattr(model_config, 'position_ablation', 'none'))
                model_config.position_ablation_seed = getattr(runtime_config, 'position_ablation_seed', getattr(model_config, 'position_ablation_seed', 0))
                model_config.position_ablation_ref_index = getattr(runtime_config, 'position_ablation_ref_index', getattr(model_config, 'position_ablation_ref_index', 0))
                model_config.position_ablation_virtual_coord = getattr(runtime_config, 'position_ablation_virtual_coord', getattr(model_config, 'position_ablation_virtual_coord', None))
                model_config.spatial_depth = getattr(runtime_config, 'spatial_depth', getattr(model_config, 'spatial_depth', None))
                model_config.vit_order = getattr(runtime_config, 'vit_order', getattr(model_config, 'vit_order', 'spatial_temporal'))
            model = create_model_from_config(model_config, num_voxels)
        
            model.load_checkpoint(metafile['model'])
        else:
            # model = eeg_encoder_ours(time_len=num_voxels)
            position_flag = getattr(runtime_config, 'position_flag', True) if runtime_config is not None else True
            mask_ratio = getattr(runtime_config, 'mask_ratio', 0.75) if runtime_config is not None else 0.75
            time_len = getattr(runtime_config, 'eeg_data_len', 512) if runtime_config is not None else 512
            in_chans = getattr(runtime_config, 'eeg_data_chan', 128) if runtime_config is not None else 128
            depth = getattr(runtime_config, 'depth', 24) if runtime_config is not None else 24
            num_heads = getattr(runtime_config, 'num_heads', 16) if runtime_config is not None else 16
            xyz_brain = getattr(runtime_config, 'xyz_brain', '/home/mahui/Dataset/EEGCVPR40/XYZ_Brain_EEGCVPR128.pth') if runtime_config is not None else '/home/mahui/Dataset/EEGCVPR40/XYZ_Brain_EEGCVPR128.pth'
            fixed_electrode_indices = getattr(runtime_config, 'fixed_electrode_indices', None) if runtime_config is not None else None
            fixed_electrode_index_base = getattr(runtime_config, 'fixed_electrode_index_base', 0) if runtime_config is not None else 0
            position_ablation = getattr(runtime_config, 'position_ablation', 'none') if runtime_config is not None else 'none'
            position_ablation_seed = getattr(runtime_config, 'position_ablation_seed', 0) if runtime_config is not None else 0
            position_ablation_ref_index = getattr(runtime_config, 'position_ablation_ref_index', 0) if runtime_config is not None else 0
            position_ablation_virtual_coord = getattr(runtime_config, 'position_ablation_virtual_coord', None) if runtime_config is not None else None
            spatial_depth = getattr(runtime_config, 'spatial_depth', None) if runtime_config is not None else None
            vit_order = getattr(runtime_config, 'vit_order', 'spatial_temporal') if runtime_config is not None else 'spatial_temporal'
            model = eeg_encoder_ours_V4(time_len=time_len,in_chans=in_chans,depth=depth,num_heads=num_heads,
                                        mask_ratio=mask_ratio,position_flag=position_flag,xyz_brain=xyz_brain,
                                        fixed_electrode_indices=fixed_electrode_indices,
                                        fixed_electrode_index_base=fixed_electrode_index_base,
                                        position_ablation=position_ablation,
                                        position_ablation_seed=position_ablation_seed,
                                        position_ablation_ref_index=position_ablation_ref_index,
                                        position_ablation_virtual_coord=position_ablation_virtual_coord,
                                        spatial_depth=spatial_depth,
                                        vit_order=vit_order)
            print('无预训练模式，掩码率:',mask_ratio)
            print('开启位置编码：',position_flag)

        self.mae = model
        if clip_tune:
            self.mapping = mapping(input_dim=model.embed_dim, seq_len=model.num_patches)
        dataset_name = getattr(runtime_config, 'dataset', '') if runtime_config is not None else ''
        self.is_things_eeg2 = dataset_name in ['THINGS-EEG2', 'THINGS_EEG2', 'THINGSEEG2']
        self.things_clip_loss = getattr(runtime_config, 'things_clip_loss', 'cosine')
        self.things_clip_temperature = float(
            getattr(runtime_config, 'things_clip_temperature', 0.07))
        self.things_clip_cosine_weight = float(
            getattr(runtime_config, 'things_clip_cosine_weight', 1.0))
        self.things_clip_contrastive_weight = float(
            getattr(runtime_config, 'things_clip_contrastive_weight', 0.1))
        self.things_clip_queue_size = int(
            getattr(runtime_config, 'things_clip_queue_size', 0))
        self.things_clip_context_weight = float(
            getattr(runtime_config, 'things_clip_context_weight', 0.0))
        self.things_freeze_eeg_encoder = bool(
            getattr(runtime_config, 'things_freeze_eeg_encoder', False))
        self.things_eeg_encoder_lr_scale = float(
            getattr(runtime_config, 'things_eeg_encoder_lr_scale', 1.0))
        if self.is_things_eeg2 and self.things_freeze_eeg_encoder:
            for param in self.mae.parameters():
                param.requires_grad = False
        self.clip_loss_details = {}
        if (self.is_things_eeg2
                and self.things_clip_loss == 'cosine_contrastive'
                and self.things_clip_queue_size > 0):
            self.register_buffer(
                '_things_image_queue',
                torch.zeros(self.things_clip_queue_size, 768),
                persistent=False)
            self.register_buffer(
                '_things_queue_ptr', torch.zeros(1, dtype=torch.long), persistent=False)
            self.register_buffer(
                '_things_queue_filled', torch.zeros(1, dtype=torch.long), persistent=False)
        if self.is_things_eeg2:
            if clip_tune and self.things_clip_context_weight != 0:
                self.things_clip_context_norm = nn.LayerNorm(cond_dim)
            print('THINGS-EEG2 CLIP loss:', self.things_clip_loss,
                  'temperature:', self.things_clip_temperature,
                  'cosine_weight:', self.things_clip_cosine_weight,
                  'contrastive_weight:', self.things_clip_contrastive_weight,
                  'queue_size:', self.things_clip_queue_size,
                  'context_weight:', self.things_clip_context_weight,
                  'freeze_eeg_encoder:', self.things_freeze_eeg_encoder,
                  'eeg_encoder_lr_scale:', self.things_eeg_encoder_lr_scale)
        if cls_tune:
            self.cls_net = classify_network()

        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        # self.fmri_latent_dim = model.masked_channels#encoder输出为mask之后的通道数
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

        # self.image_embedder = FrozenImageEmbedder()

    def refresh_position_encoding(self):
        if hasattr(self.mae, 'refresh_position_encoding'):
            self.mae.refresh_position_encoding()

    # def forward(self, x):
    #     # n, c, w = x.shape
    #     latent_crossattn = self.mae(x)
    #     if self.global_pool == False:
    #         latent_crossattn = self.channel_mapper(latent_crossattn)
    #     latent_crossattn = self.dim_mapper(latent_crossattn)
    #     out = latent_crossattn
    #     return out

    def forward(self, x):
        # n, c, w = x.shape
        freeze_eeg_encoder = self.is_things_eeg2 and self.things_freeze_eeg_encoder
        if freeze_eeg_encoder:
            self.mae.eval()
            with torch.no_grad():
                latent_crossattn = self.mae(x)
        else:
            latent_crossattn = self.mae(x)
        latent_return = latent_crossattn
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        if hasattr(self, 'things_clip_context_norm'):
            global_context = self.things_clip_context_norm(
                self.mapping(latent_return)).unsqueeze(1)
            out = out + self.things_clip_context_weight * global_context
        return out, latent_return

    # def recon(self, x):
    #     recon = self.decoder(x)
    #     return recon

    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        target_emb = self.mapping(x)
        cosine = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        self.clip_loss_details = {}
        if not (self.is_things_eeg2 and self.things_clip_loss == 'cosine_contrastive'):
            return cosine

        eeg_features = F.normalize(target_emb, dim=-1)
        image_features = F.normalize(image_embeds, dim=-1)
        batch_size = eeg_features.shape[0]
        labels = torch.arange(batch_size, device=eeg_features.device)

        image_candidates = image_features
        queue_filled = 0
        if hasattr(self, '_things_image_queue'):
            queue_filled = int(self._things_queue_filled.item())
            if queue_filled > 0:
                queued_images = self._things_image_queue[:queue_filled].to(
                    device=image_features.device, dtype=image_features.dtype)
                image_candidates = torch.cat([image_features, queued_images], dim=0)

        temperature = max(self.things_clip_temperature, 1e-6)
        eeg_to_image_logits = eeg_features @ image_candidates.t() / temperature
        image_to_eeg_logits = image_features @ eeg_features.t() / temperature
        eeg_to_image_loss = F.cross_entropy(eeg_to_image_logits, labels)
        image_to_eeg_loss = F.cross_entropy(image_to_eeg_logits, labels)
        contrastive = 0.5 * (eeg_to_image_loss + image_to_eeg_loss)
        loss = (self.things_clip_cosine_weight * cosine
                + self.things_clip_contrastive_weight * contrastive)

        with torch.no_grad():
            eeg_to_image_acc = (eeg_to_image_logits.argmax(dim=1) == labels).float().mean()
            image_to_eeg_acc = (image_to_eeg_logits.argmax(dim=1) == labels).float().mean()
            self.clip_loss_details = {
                'loss_clip_cosine': cosine.detach(),
                'loss_clip_contrastive': contrastive.detach(),
                'clip_retrieval_eeg_to_image': eeg_to_image_acc,
                'clip_retrieval_image_to_eeg': image_to_eeg_acc,
                'clip_queue_filled': torch.tensor(
                    float(queue_filled), device=eeg_features.device),
            }
            if self.training and hasattr(self, '_things_image_queue'):
                self._enqueue_things_images(image_features.detach())
        return loss

    @torch.no_grad()
    def _enqueue_things_images(self, image_features):
        queue_size = self._things_image_queue.shape[0]
        if queue_size == 0:
            return
        if image_features.shape[0] >= queue_size:
            self._things_image_queue.copy_(image_features[-queue_size:])
            self._things_queue_ptr.zero_()
            self._things_queue_filled.fill_(queue_size)
            return

        ptr = int(self._things_queue_ptr.item())
        count = image_features.shape[0]
        first_count = min(count, queue_size - ptr)
        self._things_image_queue[ptr:ptr + first_count].copy_(
            image_features[:first_count])
        remaining = count - first_count
        if remaining > 0:
            self._things_image_queue[:remaining].copy_(image_features[first_count:])
        self._things_queue_ptr.fill_((ptr + count) % queue_size)
        self._things_queue_filled.fill_(min(
            queue_size, int(self._things_queue_filled.item()) + count))
    


class eLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune=True, cls_tune=False,
                 runtime_config=None):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.ckp_path = os.path.join(pretrain_root, 'models/v1-5-pruned.ckpt')
        self.config_path = os.path.join(pretrain_root, 'models/config15.yaml') 
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
       
        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool,
                                                  clip_tune=clip_tune, cls_tune=cls_tune,
                                                  runtime_config=runtime_config)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            print('启用日志')
            #logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune
        if not clip_tune and hasattr(self.model, 'image_embedder'):
            del self.model.image_embedder

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    @staticmethod
    def _print_parameter_count(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print('联合训练阶段模型参数量:')
        print(f'  total params: {total_params:,} ({total_params / 1e6:.3f}M)')
        print(f'  trainable params: {trainable_params:,} ({trainable_params / 1e6:.3f}M)')
        print(f'  frozen params: {frozen_params:,} ({frozen_params / 1e6:.3f}M)')

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        # dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs1, shuffle=False)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()
        self._print_parameter_count(self.model)

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                print(item)
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)



class eLDM_eval:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune=True, cls_tune=False,
                 runtime_config=None):

        self.config_path = os.path.join(pretrain_root, 'models/config15.yaml')
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)

        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool,
                                                  clip_tune=clip_tune, cls_tune=cls_tune,
                                                  runtime_config=runtime_config)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device
        self.model = model

        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                 output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one

        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs1, shuffle=False)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()

        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path=None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels,
                     self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                     HW[0] // 2 ** (num_resolutions - 1), HW[1] // 2 ** (num_resolutions - 1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)

        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                # print(item)
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w')  # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'

                c, re_latent = model.get_learned_conditioning(
                    repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=num_samples,
                                                 shape=shape,
                                                 verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image + 1.0) / 2.0, min=0.0, max=1.0)

                all_samples.append(
                    torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0))  # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(
                        np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path,
                                                                 f'./test{count}-{copy_idx}.png'))

        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples + 1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')

        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
