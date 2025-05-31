import torch 
import numpy as np 
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from tqdm import tqdm
def evaluate_pre(val_dataloader, model_clip, MyModel, device, args, val_obj_list_post, PFL_TextEncoder, tokenizer, stage):
    model_clip.eval()
    MyModel.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['pr_sp'] = []
    results['gt_sp'] = []
    id = 0
    print("-------------------start evaluating ---------------------")
    val_dataloader = tqdm(val_dataloader)
    for items in val_dataloader:
        id = id + 1
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        results['gt_sp'].append(items['anomaly'].item())
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        with torch.no_grad():
            image_features, proj_patch_tokens, patch_tokens =  model_clip.encode_image(image, args.features_list)
            text_embeddings, loss_reg = MyModel.forward_ensemble(PFL_TextEncoder, image_features, patch_tokens ,cls_name, device, tokenizer, mode = "val")
            temp_cls = 0
            pro_img, anomaly_maps_list = MyModel(text_embeddings, image_features, patch_tokens, stage = stage, mode = "val")
            pro_img = pro_img.squeeze(2)
            for i in range(args.prompt_num * args.sample_num):
                text_probs = torch.cat([pro_img[:,i].unsqueeze(0), pro_img[:,i + args.prompt_num * args.sample_num].unsqueeze(0)], dim = 1).softmax(dim = -1)
                temp_cls = temp_cls + text_probs[0, 1]
            temp_cls = temp_cls / (args.prompt_num * args.sample_num)

            anomaly_maps = []
            for num in range(len(anomaly_maps_list)):
                anomaly_map = anomaly_maps_list[num]
                temp = 0
                for i in range(args.prompt_num* args.sample_num):
                    temp = temp + torch.softmax(torch.stack([anomaly_map[:,i,:,:], anomaly_map[:,i+(args.prompt_num * args.sample_num) ,:,:]], dim = 1), dim =1)
                anomaly_map =  temp[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            anomaly_map = np.sum(anomaly_maps, axis=0)[0]
            results['anomaly_maps'].append(anomaly_map)
            results['pr_sp'].append(temp_cls.cpu().numpy())

    # metrics
    ap_px_ls = []
    ap_sp_ls = []
    for obj in val_obj_list_post:
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
                gt_sp.append(results['gt_sp'][idxes])
                pr_sp.append(results['pr_sp'][idxes])
        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)
        gt_sp = np.array(gt_sp)
        pr_sp = np.array(pr_sp)
        pr_sp = (pr_sp - pr_sp.min()) / (pr_sp.max() - pr_sp.min() + 1e-8)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        ap_sp = average_precision_score(gt_sp.ravel(), pr_sp.ravel())
        ap_px_ls.append(ap_px)
        ap_sp_ls.append(ap_sp)
    ap_mean_pixel = np.mean(ap_px_ls)
    ap_mean_image = np.mean(ap_sp_ls)
    model_clip.eval()
    MyModel.train()
    del patch_tokens, results, anomaly_map
    torch.cuda.empty_cache()
    print("------------------- end evaluating ---------------------")
    return ap_mean_pixel, ap_mean_image