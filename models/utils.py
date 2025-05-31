from itertools import repeat
import collections.abc
import torch
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from tabulate import tabulate
from scipy.ndimage import gaussian_filter






def log_normal_dist(x, mean, logvar, dim):
    log_norm = -0.5 * (logvar + (x - mean) * (x - mean) * logvar.exp().reciprocal()) 

    return torch.sum(log_norm, dim)

def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)









def calcuate_metric_image(results, obj_list, logger, alpha = 0.9, sigm = 4, args = None):
    # metrics
    print(f"==================================  alpha: {alpha}")
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_px_ls = []
    aupro_sp_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    iou_list = []
    iou_list_ls = []
    table_best_the = []
    for obj in obj_list:  #具体类别名称列表
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_list = []
        img_path_list = []

        table.append(obj)
        
        if obj in ["capsules", "macaroni1", "macaroni2", "pipe_fryum", "screw", "cashew", "chewinggum"]:
            can_k = -20
        else:
            can_k = -2000
        #can_k = -1000
        #can_k = -1
        

        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].cpu().numpy())
                pr_px.append(results['anomaly_maps'][idxes])

                temp = np.partition(results['anomaly_maps'][idxes].reshape(-1), kth=can_k)[can_k:]
                pr_sp_list.append(np.mean(temp))   
                

                gt_sp.append(results['gt_sp'][idxes])  
                pr_sp.append(results['pr_sp'][idxes])  
                img_path_list.append(results['path'][idxes])
        gt_px = np.array(gt_px) 
        gt_sp = np.array(gt_sp)  
        pr_px = np.array(pr_px)  
        pr_sp = np.array(pr_sp)  
        pr_sp_tmp = np.array(pr_sp_list)

        pr_px =  gaussian_filter(pr_px, sigma=sigm,axes = (1,2)) 



        pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min() + 1e-8)
        pr_sp = (pr_sp - pr_sp.min()) / (pr_sp.max() - pr_sp.min() + 1e-8)
        pr_sp = (alpha * pr_sp + (1 - alpha) * pr_sp_tmp)
        #pr_sp = pr_sp_tmp

        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())   

        auroc_sp = roc_auc_score(gt_sp, pr_sp)  
        ap_sp = average_precision_score(gt_sp, pr_sp)   
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())


        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        best_threshold_cls = thresholds[np.argmax(f1_scores)]
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])

        # aupr
        #aupro_sp = auc(recalls, precisions)
        aupro_sp = 0
        


        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls+ 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        iou = cal_iou(gt_px.ravel(), (pr_px.ravel()>best_threshold))
        iou_list.append(iou)
        print("{}--->  iou:{}   f1-max:{}  threshold:{}".format(obj,iou,f1_px,best_threshold)) 


        
        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        #aupro_px = cal_pro_score(gt_px, pr_px)
        aupro_px = 0

        '''
        print("正在可视化{}".format(obj))    
        for i in range(len(img_path_list)):
            cls = img_path_list[i].split('/')[-2]
            filename = img_path_list[i].split('/')[-1]
            save_vis = os.path.join(args.save_path, 'imgs', obj, cls)
            vis_img = vis_img = cv2.resize(cv2.imread(img_path_list[i]), (img_size, img_size))
            visualization(save_root= save_vis, pic_name=filename, raw_image= vis_img, raw_anomaly_map= np.squeeze(pr_px[i]), raw_gt= np.squeeze(gt_px[i]), the = best_threshold)
        '''



        table.append(str(np.round(auroc_px * 100, decimals=2)))
        table.append(str(np.round(aupro_px * 100, decimals=2)))
        table.append(str(np.round(ap_px * 100, decimals=2)))

        table.append(str(np.round(f1_px * 100, decimals=2)))
        table.append(str(np.round(iou * 100, decimals=2)))


        table.append(str(np.round(auroc_sp * 100, decimals=2)))
        table.append(str(np.round(aupro_sp * 100, decimals=2)))

        table.append(str(np.round(ap_sp * 100, decimals=2)))
        table.append(str(np.round(f1_sp * 100, decimals=2)))
        table.append(str(np.round(best_threshold, decimals=3)))
        

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_px_ls.append(aupro_px)
        aupro_sp_ls.append(aupro_sp)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)
        iou_list_ls.append(iou)
        table_best_the.append(best_threshold)

    # logger
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=2)),
                     str(np.round(np.mean(aupro_px_ls) * 100, decimals=2)),
                      str(np.round(np.mean(ap_px_ls) * 100, decimals=2)),
                      str(np.round(np.mean(f1_px_ls) * 100, decimals=2)), 
                      str(np.round(np.mean(iou_list_ls) * 100, decimals=2)), 
                      str(np.round(np.mean(auroc_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(aupro_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(ap_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(f1_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(table_best_the), decimals=3))])
    
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'aupro_px', 'ap_px', 'f1_px', 'iou',"auroc_sp","aupro_sp","ap_sp", "f1_sp", "threshold"], tablefmt="pipe")
    import pandas as pd
    headers = ['objects', 'auroc_px', 'aupro_px', 'ap_px', 'f1_px', 'iou', "auroc_sp", "aupro_sp", "ap_sp", "f1_sp", "threshold"]
    df = pd.DataFrame(table_ls, columns=headers)
    csv_file_path = f'./results.csv'
    df.to_csv(csv_file_path, index=False)

    logger.info("\n%s", results)
    print(args.checkpoint_path, args.sample_num)
