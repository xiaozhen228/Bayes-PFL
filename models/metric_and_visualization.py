import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from tabulate import tabulate
from scipy.ndimage import gaussian_filter
import pandas as pd
import os 
import cv2
from skimage import measure

datasets_name_list = ["mvtec","visa","BSD","RSDD", "KSDD2", "ISIC", "ClinicDB", "ColonDB", "HeadCT", "TN3K", "DTD", "Endo", "BrainMRI", "Br35H", "Kvasir", "BTAD"]

# Only for segmentation
datasets_no_good = ["ISIC", "ClinicDB", "ColonDB", "Kvasir", "Endo", "BSD", "RSDD", "KSDD2", "TN3K"]

# Only for classification
datasets_class = ["HeadCT", "BrainMRI", "Br35H"]

def cal_iou(gt,pre):
    ground_truth = gt.astype(np.uint8)
    prediction = pre.astype(np.uint8)
    intersection = np.logical_and(prediction, ground_truth)
    union = np.logical_or(prediction, ground_truth)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def normalize(pred, max_value=None, min_value=None):

    if max_value is None or min_value is None:
        if (pred.max() - pred.min()) == 0:
            return np.zeros_like(pred)
        else:
            return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def he_cheng(img_list, size = 256):
    h,w,c = img_list[0].shape
    jian = np.ones((h, 10, 3),dtype=np.uint8) * 255
    vis_con = img_list[0]
    for i in range(1,len(img_list)):
        vis_con = np.concatenate([vis_con, jian, img_list[i]], axis=1)

    vis_con = cv2.resize(vis_con, (size*len(img_list)+ 10*(len(img_list)-1), size)).astype(np.uint8)
    return vis_con


def visualization(save_root, pic_name, raw_image, raw_anomaly_map, raw_gt, the = 0.5, size = 518):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    save_npy = save_root.replace("imgs", "npy")

    if not os.path.exists(save_npy):
        os.makedirs(save_npy)

    
    

    
    assert len(raw_image.shape) == 3 and len(raw_anomaly_map.shape) == 2 and len(raw_gt.shape) == 2
    map = raw_anomaly_map
    gt = raw_gt

    #np.save(os.path.join(save_npy, "text_"+pic_name.replace('bmp', 'npy')), text)
    #np.save(os.path.join(save_npy, "vis_map_"+pic_name.replace('bmp', 'npy')), map)
    #np.save(os.path.join(save_npy, "gt_"+pic_name.replace('bmp', 'npy')), gt)

    
    
    img = cv2.cvtColor(raw_image , cv2.COLOR_BGR2RGB)
    map = normalize(raw_anomaly_map)
    gt = normalize(raw_gt)
    map_binary = np.array(raw_anomaly_map> the, dtype= np.uint8)
    map_crop = map * map_binary

    ground_truth_contours, _ = cv2.findContours(np.array(raw_gt * 255, dtype = np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis_map = apply_ad_scoremap(img, map)
    vis_gt = apply_ad_scoremap(img, gt)
    vis_map_binary = apply_ad_scoremap(img, map_binary)
    vis_map_crop = apply_ad_scoremap(img, map_crop)

    vis_map = cv2.cvtColor(vis_map, cv2.COLOR_RGB2BGR)
    vis_gt = cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR)
    vis_map_binary = cv2.cvtColor(vis_map_binary, cv2.COLOR_RGB2BGR)
    vis_map_crop = cv2.cvtColor(vis_map_crop, cv2.COLOR_RGB2BGR)

    vis_map_binary = cv2.drawContours(vis_map_binary, ground_truth_contours, -1, (0, 255, 0), 2)
    vis_map_crop = cv2.drawContours(vis_map_crop, ground_truth_contours, -1, (0, 255, 0), 2)  

    zong = he_cheng([raw_image, vis_map, vis_map_crop, vis_gt])
    #cv2.imwrite(os.path.join(save_root, "vis_map_"+pic_name), vis_map)
    #cv2.imwrite(os.path.join(save_root, "vis_gt_"+pic_name), vis_gt)
    #cv2.imwrite(os.path.join(save_root, "vis_map_binary_"+pic_name), vis_map_binary)
    #cv2.imwrite(os.path.join(save_root, "vis_map_crop_"+pic_name), vis_map_crop)
    cv2.imwrite(os.path.join(save_root, "vis_zong_"+pic_name.replace('bmp', 'png')), zong)



def calcuate_metric_pixel(results, obj_list, logger, alpha = 0.9, sigm = 4, args = None):
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
    for obj in obj_list:
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

        if sigm != 0:
            pr_px =  gaussian_filter(pr_px, sigma=sigm,axes = (1,2)) 



        pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min() + 1e-8)
        pr_sp = (pr_sp - pr_sp.min()) / (pr_sp.max() - pr_sp.min() + 1e-8)
        pr_sp = (alpha * pr_sp + (1 - alpha) * pr_sp_tmp)
        #pr_sp = pr_sp_tmp

        All_anomaly = (np.sum(gt_sp) == pr_px.shape[0])

        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel()) 
        #auroc_px = 0
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())


        if All_anomaly:
            auroc_sp  = 0
            ap_sp = 0
            f1_sp = 0
            aupro_sp = 0
        
        else:

            auroc_sp = roc_auc_score(gt_sp, pr_sp)  
            ap_sp = average_precision_score(gt_sp, pr_sp) 
            # f1_sp
            precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
            f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-6)
            best_threshold_cls = thresholds[np.argmax(f1_scores)]
            f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
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
        gt_px = gt_px.squeeze()
        pr_px = pr_px.squeeze()
        #aupro_px = cal_pro_score(gt_px, pr_px)
        aupro_px = 0

        
        print("Visualization {}".format(obj))
        for i in range(len(img_path_list)):
            cls = img_path_list[i].split('/')[-2]
            filename = img_path_list[i].split('/')[-1]
            save_vis = os.path.join(args.save_path, 'imgs', obj, cls)
            vis_img = vis_img = cv2.resize(cv2.imread(img_path_list[i]), (args.image_size, args.image_size))
            visualization(save_root= save_vis, pic_name=filename, raw_image= vis_img, raw_anomaly_map= np.squeeze(pr_px[i]), raw_gt= np.squeeze(gt_px[i]), the = best_threshold)
        
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
    headers = ['objects', 'auroc_px', 'aupro_px', 'ap_px', 'f1_px', 'iou', "auroc_sp", "aupro_sp", "ap_sp", "f1_sp", "threshold"]
    df = pd.DataFrame(table_ls, columns=headers)
    csv_file_path = f'./{args.save_path}/results_{args.dataset}.csv'
    df.to_csv(csv_file_path, index=False)
    logger.info("\n%s", results)
    logger.info("\n%s", args.checkpoint_path)
    logger.info("\n%s", args.sample_num)


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
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_list = []
        img_path_list = []

        table.append(obj)
        
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_sp.append(results['gt_sp'][idxes])  
                pr_sp.append(results['pr_sp'][idxes]) 
                img_path_list.append(results['path'][idxes])

        gt_sp = np.array(gt_sp)
        pr_sp = np.array(pr_sp)

        pr_sp = (pr_sp - pr_sp.min()) / (pr_sp.max() - pr_sp.min() + 1e-8)


        auroc_px = 0
        ap_px = 0

        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp) 
        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        best_threshold_cls = thresholds[np.argmax(f1_scores)]
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        #aupro_sp = auc(recalls, precisions)
        aupro_sp = 0


        # f1_px
        f1_px = 0
        iou = 0
        best_threshold = 0.5
        iou_list.append(iou)

        aupro_px = 0
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
    headers = ['objects', 'auroc_px', 'aupro_px', 'ap_px', 'f1_px', 'iou', "auroc_sp", "aupro_sp", "ap_sp", "f1_sp", "threshold"]
    df = pd.DataFrame(table_ls, columns=headers)
    csv_file_path = f'./{args.save_path}/results_{args.dataset}.csv'
    df.to_csv(csv_file_path, index=False)
    logger.info("\n%s", results)
    logger.info("\n%s", args.checkpoint_path)
    logger.info("\n%s", args.sample_num)