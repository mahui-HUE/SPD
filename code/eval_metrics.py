from os import get_inheritable
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import ViT_H_14_Weights, vit_h_14
import torch
from einops import rearrange
from torchmetrics.functional import accuracy
from PIL import Image
from scipy.stats import combine_pvalues, ttest_1samp

def larger_the_better(gt, comp):
    return gt > comp

def smaller_the_better(gt, comp):
    return gt < comp

def mse_metric(img1, img2):
    return (np.square(img1 - img2)).mean()

def pcc_metric(img1, img2):
    return np.corrcoef(img1.reshape(-1), img2.reshape(-1))[0, 1]

def ssim_metric(img1, img2):
    return ssim(img1, img2, data_range=255, channel_axis=-1)

def identity(x):
    return x

class psm_wrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)

    @torch.no_grad()
    def __call__(self, img1, img2):
        if img1.shape[-1] == 3:
            img1 = rearrange(img1, 'w h c -> c w h')
            img2 = rearrange(img2, 'w h c -> c w h')
        img1 = img1 / 127.5 - 1.0
        img2 = img2 / 127.5 - 1.0
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        return self.lpips(torch.FloatTensor(img1).to(self.device), torch.FloatTensor(img2).to(self.device)).item()

class fid_wrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fid = FrechetInceptionDistance(feature=64)

    @torch.no_grad()
    def __call__(self, pred_imgs, gt_imgs):
        self.fid.reset()
        self.fid.update(torch.tensor(rearrange(gt_imgs, 'n w h c -> n c w h')), real=True)
        self.fid.update(torch.tensor(rearrange(pred_imgs, 'n w h c -> n c w h')), real=False)
        return self.fid.compute().item() 

def pair_wise_score(pred_imgs, gt_imgs, metric, is_sucess):
    # pred_imgs: n, w, h, 3
    # gt_imgs: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    assert len(pred_imgs) == len(gt_imgs)
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    assert isinstance(metric, fid_wrapper) == False, 'FID not supported'
    corrects = []
    for idx, pred in enumerate(pred_imgs):
        gt = gt_imgs[idx]
        gt_score = metric(pred, gt)
        rest = [img for i, img in enumerate(gt_imgs) if i != idx]
        count = 0
        for comp in rest:
            comp_score = metric(pred, comp)
            if is_sucess(gt_score, comp_score):
                count += 1
        corrects.append(count / len(rest))
    return corrects

def n_way_scores(pred_imgs, gt_imgs, metric, is_sucess, n=2, n_trials=100):
    # pred_imgs: n, w, h, 3
    # gt_imgs: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    assert len(pred_imgs) == len(gt_imgs)
    assert n <= len(pred_imgs) and n >= 2
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    assert isinstance(metric, fid_wrapper) == False, 'FID not supported'
    corrects = []
    for idx, pred in enumerate(pred_imgs):
        gt = gt_imgs[idx]
        gt_score = metric(pred, gt)
        rest = np.stack([img for i, img in enumerate(gt_imgs) if i != idx])
        correct_count = 0
        for _ in range(n_trials):
            n_imgs_idx = np.random.choice(len(rest), n-1, replace=False)
            n_imgs = rest[n_imgs_idx]
            count = 0
            for comp in n_imgs:
                comp_score = metric(pred, comp)
                if is_sucess(gt_score, comp_score):
                    count += 1
            if count == len(n_imgs):
                correct_count += 1 
        corrects.append(correct_count / n_trials)
    return corrects

def metrics_only(pred_imgs, gt_imgs, metric, *args, **kwargs):
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0

    return metric(pred_imgs, gt_imgs)

@torch.no_grad()
def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1, return_trials=False):
    pick_range =[i for i in np.arange(len(pred)) if i != class_id]
    acc_list = []
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])
        acc = accuracy(pred_picked.unsqueeze(0), torch.tensor([0], device=pred.device), 
                    top_k=top_k)
        acc_list.append(acc.item())
    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list, ddof=1) if len(acc_list) > 1 else 0.0
    if return_trials:
        return mean_acc, std_acc, acc_list
    return mean_acc, std_acc

def one_sample_greater_pvalue(values, chance_level):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return np.nan
    if np.allclose(values, values[0]):
        return 0.0 if values[0] > chance_level else 1.0
    return ttest_1samp(values, chance_level, alternative='greater').pvalue

@torch.no_grad()
def get_n_way_top_k_acc(pred_imgs, ground_truth, n_way, num_trials, top_k, device, return_std=False, return_stats=False):
    weights = ViT_H_14_Weights.DEFAULT
    model = vit_h_14(weights=weights)
    preprocess = weights.transforms()
    model = model.to(device)
    model = model.eval()
    
    acc_list = []
    std_list = []
    p_list = []
    trial_list = []
    chance_level = top_k / n_way
    for pred, gt in zip(pred_imgs, ground_truth):
        pred = preprocess(Image.fromarray(pred.astype(np.uint8))).unsqueeze(0).to(device)
        gt = preprocess(Image.fromarray(gt.astype(np.uint8))).unsqueeze(0).to(device)
        gt_class_id = model(gt).squeeze(0).softmax(0).argmax().item()
        pred_out = model(pred).squeeze(0).softmax(0).detach()

        acc, std, trials = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k, return_trials=True)
        acc_list.append(acc)
        std_list.append(std)
        p_list.append(one_sample_greater_pvalue(trials, chance_level))
        trial_list.extend(trials)
       
    if return_stats:
        valid_p = [p for p in p_list if not np.isnan(p)]
        combined_p = combine_pvalues(valid_p, method='fisher').pvalue if len(valid_p) > 0 else np.nan
        return {
            'acc': acc_list,
            'std': std_list,
            'p': p_list,
            'overall_std': np.std(trial_list, ddof=1) if len(trial_list) > 1 else 0.0,
            'overall_p': one_sample_greater_pvalue(trial_list, chance_level),
            'combined_p': combined_p,
            'chance_level': chance_level,
        }
    if return_std:
        return acc_list, std_list
    return acc_list

def get_similarity_metric(img1, img2, method='pair-wise', metric_name='mse', **kwargs):
    # img1: n, w, h, 3
    # img2: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    if img1.shape[-1] != 3:
        img1 = rearrange(img1, 'n c w h -> n w h c')
    if img2.shape[-1] != 3:
        img2 = rearrange(img2, 'n c w h -> n w h c')

    if method == 'pair-wise':
        eval_procedure_func = pair_wise_score 
    elif method == 'n-way':
        eval_procedure_func = n_way_scores
    elif method == 'metrics-only':
        eval_procedure_func = metrics_only
    elif method == 'class':
        return get_n_way_top_k_acc(img1, img2, **kwargs)
    else:
        raise NotImplementedError

    if metric_name == 'mse':
        metric_func = mse_metric
        decision_func = smaller_the_better
    elif metric_name == 'pcc':
        metric_func = pcc_metric
        decision_func = larger_the_better
    elif metric_name == 'ssim':
        metric_func = ssim_metric
        decision_func = larger_the_better
    elif metric_name == 'psm':
        metric_func = psm_wrapper()
        decision_func = smaller_the_better
    elif metric_name == 'fid':
        metric_func = fid_wrapper()
        decision_func = smaller_the_better
    else:
        raise NotImplementedError
    
    return eval_procedure_func(img1, img2, metric_func, decision_func, **kwargs)
