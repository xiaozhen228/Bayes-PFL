import os
import  json 
import argparse 
import numpy as np 
import random 
import os
import torch
from torch import nn
from torch.nn import functional as F 
import logging
from datasets import Makedataset
from loss import FocalLoss, DiceLoss, Orthogonal_Loss
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from models.VPB import Context_Prompting
from models.VPB import TextEncoder
from models.evaluate import evaluate_pre

import open_clip_local as open_clip
from models.EMA import EMA
from models.model_CLIP import Load_CLIP, tokenize
import sys

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
def _convert_image_to_rgb(image):
    return image.convert("RGB")
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _transform_test(n_px):
    return Compose([
        Resize((n_px,n_px), interpolation=BICUBIC),
        CenterCrop((n_px,n_px)),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def train(args):
    image_size = args.image_size 
    learning_rate = args.learning_rate 
    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    log_path = os.path.join(save_path,"result.txt")
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    args.vision_width = model_configs["vision_cfg"]['width']
    args.text_width = model_configs['text_cfg']['width']
    args.embed_dim = model_configs['embed_dim']
    

    # We retained the OpenCLIP interface to enable Bayes-PFL to support a broader range of backbones.
    # -------------------------------------------------------------------------------------------------
    
    # Example 1 : The pretrained model from huggingface laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
    '''
    model_clip, _, _ = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", img_size= image_size) 
    tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
    model_clip = model_clip.to(device)
    model_clip.eval()
    '''

    # Example 2 : The pretrained model from OpenAI CLIP
    '''
    model_clip, _, _ = open_clip.create_model_and_transforms(args.model, pretrained= args.pretrained, img_size= image_size) 
    tokenizer = open_clip.get_tokenizer(args.model)
    model_clip = model_clip.to(device)
    model_clip.eval()
    '''
    
    # -------------------------------------------------------------------------------------------------
    
    # This is from our own implementation of the CLIP model, which only supports the OpenAI pretrained models ViT-B-16, ViT-L-14, and ViT-L-14-336.
    model_clip , _ , _ = Load_CLIP(image_size, args.pretrained_path , device=device) 
    tokenizer = tokenize
    model_clip.to(device)
    model_clip.eval()

    # log 
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    
    logger.setLevel(logging.INFO)
    file_hander  = logging.FileHandler(log_path, mode = 'a+')
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)
    console_hander = logging.StreamHandler()
    console_hander.setFormatter(formatter)
    logger.addHandler(console_hander)


    # 记录参数 
    for arg in vars(args): 
        logger.info(f'{arg}: {getattr(args,arg)}')

    preprocess_test = _transform_test(image_size)


    Make_dataset = Makedataset(train_data_path = args.train_data_path , preprocess_test = preprocess_test, mode = "train", 
                               image_size = args.image_size)

    Make_dataset_val_post = Makedataset(train_data_path = args.val_data_path , preprocess_test = preprocess_test, mode = "test", 
                               image_size = args.image_size)
    

    PFL_TextEncoder = TextEncoder(model_clip, args)
    MyModel = Context_Prompting(args = args).to(device)
    MyModel.train()


    parameter_prompt_list = []
    parameter_prompt2_list = []
    parameter_linear_list = []
    parameter_linear_list_class = []

    for n, m in MyModel.named_parameters():
        if "prompt" in n:
            parameter_prompt_list.append(m)
        elif "PFL" in n and "encoder" not in n and "decoder" not in n:
            parameter_prompt2_list.append(m)
        elif "temperature" not in n:
            parameter_linear_list.append(m)
        else:
            print(n)
  
    for n, m in MyModel.named_parameters():
        if "class_mapping" in n or "image_mapping" in n or "fuse" in n:
            print(n)
            parameter_linear_list_class.append(m)



    lr_group1 = parameter_prompt_list
    lr_group2 = [MyModel.temperature_pixel, MyModel.temperature_image]
    lr_group3 = parameter_prompt2_list
    lr_group4 = parameter_linear_list
    lr_group5 = parameter_linear_list_class
    optimizer_stage1 = torch.optim.Adam([{'params':lr_group1, 'lr': args.learning_rate_prompt}, {'params':lr_group2, 'lr':0.01}, {'params':lr_group3, 'lr':args.learning_rate_PFL}, {'params':lr_group4, 'lr':args.learning_rate_linear}], lr = learning_rate, betas = (0.5 , 0.999)) 
    optimizer_stage2 = torch.optim.Adam([{'params':lr_group5, 'lr':args.learning_rate_linear}], lr = learning_rate, betas = (0.5 , 0.999)) 

    loss_dice_function = DiceLoss()
    loss_focal_function = FocalLoss()
    loss_ort_function = Orthogonal_Loss()
    loss_cross = nn.CrossEntropyLoss()

    epochs = args.epochs
    ema = EMA(MyModel, decay= args.alpha)
    ema.register()
    post_dataloader, post_obj_list = Make_dataset.mask_dataset(name = args.dataset, batchsize=args.batch_size, product_list= None, shuf= True)
    if args.dataset == "mvtec":
        val_dataset_name = "visa"
        val_product_list  = ["chewinggum", "cashew", "pipe_fryum","capsules", "candle"] 
    elif args.dataset == "visa":
        val_dataset_name = "mvtec"
        val_product_list  = ["bottle", "hazelnut", "wood", "zipper", "leather"]
    else:
        val_dataset_name = "mvtec"
        val_product_list  = val_product_list  = ["bottle", "hazelnut", "wood", "zipper", "leather"]
    val_post_dataloader, val_obj_list_post = Make_dataset_val_post.mask_dataset(name = val_dataset_name, product_list= val_product_list, batchsize = 1, shuf= False)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer_stage1,
        num_warmup_steps=int(0.1 * len(post_dataloader) * args.epochs),
        num_training_steps=len(post_dataloader) * args.epochs,
    )

    ap_max_pixel = 0
    ap_max_image = 0
    early_stop_patience = 3
    stage = 1
    patience = 0
    if stage == 1:
        optimizer = optimizer_stage1
    else:
        optimizer = optimizer_stage2

    for epoch_post in range(epochs):
        loss_class_list = []
        loss_dist_reg_list = []
        loss_text_list = []
        loss_seg_list = []
        idx = 0
        train_bar = tqdm(post_dataloader)
        for items in train_bar:
            optimizer.zero_grad()
            idx += 1
            image = items['img'].to(device)
            cls_name = items['cls_name']
            anomaly = items['anomaly'].to(device)
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5], gt[gt < 0.5] = 1, 0
            

            with torch.no_grad():
                image_features,  proj_patch_tokens , patch_tokens =  model_clip.encode_image(image, args.features_list)
            text_embeddings, loss_reg = MyModel.forward_ensemble(PFL_TextEncoder, image_features, patch_tokens ,cls_name, device, tokenizer, mode = "train")
            pro_img, anomaly_maps_list  = MyModel(text_embeddings, image_features, patch_tokens, stage = stage, mode = "train")
            pro_img = pro_img.squeeze(2)

            # We optimize only one pair of learnable prompts from the prompt bank at a time in order to: 1) increase the diversity of prompts in the prompt bank, and 2) reduce the cost of training.
            index_prompt = torch.randint(0,args.prompt_num, (1,1))[0]  # 
            print(index_prompt)
            pro_img = torch.cat([pro_img[:,index_prompt], pro_img[:,index_prompt + args.prompt_num]], dim = 1)
            loss_class = loss_cross(pro_img, anomaly)
            loss_dist_reg = loss_reg[1] + loss_reg[2] + loss_reg[3]
            loss_text  = loss_ort_function(text_embeddings, args)

            loss_foc = 0
            loss_dic = 0
            for num in range(len(anomaly_maps_list)):
                anomaly_map = anomaly_maps_list[num]
                anomaly_map_select = torch.cat([anomaly_map[:,index_prompt,:,:], anomaly_map[:,index_prompt + args.prompt_num,:,:]], dim = 1)
                anomaly_map_select = torch.softmax(anomaly_map_select, dim =1)
                loss_foc += loss_focal_function(anomaly_map_select.clone(), gt.clone())
                loss_dic += loss_dice_function(anomaly_map_select, gt)
            loss_seg = loss_foc + loss_dic 
            if stage == 1:  # The expression (loss_class +  loss_seg + loss_dist_reg) corresponds to the Prompt Flow Loss in Equation (7).
                loss = loss_class +  loss_seg + loss_dist_reg + loss_text
            else:
                loss = loss_class  # We found that training the classification network alone for a few epochs at the end can achieve better zero-shot classification performance.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # for i, group in enumerate(optimizer_stage1.param_groups):
            #     print(f"Group {i} LR: {group['lr']:.6f}")
            loss_class_list.append(loss_class.item())
            loss_dist_reg_list.append(loss_dist_reg.item())
            loss_text_list.append(loss_text.item())
            loss_seg_list.append(loss_seg.item())
            print(f"Segmentation Loss: {loss_seg.item()}, Classification Loss: {loss_class.item()}, Distribution Loss: {loss_dist_reg.item()}, Orthogonal_Loss: {loss_text.item()}")
            del patch_tokens, proj_patch_tokens, image_features, text_embeddings, anomaly_maps_list
            torch.cuda.empty_cache()
        if (epoch_post + 1) > 0:
            MyModel.zk_n = None
            MyModel.zk_a = None
            ap_raw_pixel, ap_raw_image = evaluate_pre(val_post_dataloader, model_clip, MyModel, device, args, val_obj_list_post, PFL_TextEncoder, tokenizer, stage)

        if (epoch_post + 1) % args.print_freq == 0:
            logger.info('epoch post [{}], loss_seg:{:.4f} loss_cls:{:.4f} loss_dist_reg:{:.4f}  loss_text:{:.4f} ap_raw_pixel:{:.4f}  ap_raw_image:{:.4f}'.format(epoch_post + 1 , np.mean(loss_seg_list), np.mean(loss_class_list),np.mean(loss_dist_reg_list),np.mean(loss_text_list), ap_raw_pixel, ap_raw_image))

        for i, group in enumerate(optimizer_stage1.param_groups):
            logger.info(f"Group {i} LR: {group['lr']:.6f}")

        
        if (epoch_post + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_post_' + str(epoch_post + 1)+f"_{stage}" + '.pth')
            save_dict = {'MyModel': MyModel.state_dict()}
            torch.save(save_dict, ckp_path)
        

        if stage == 1:
            if ap_raw_pixel >= ap_max_pixel:
                patience = 0
                ap_max_pixel = ap_raw_pixel
                logger.info('epoch post [{}/{}], ap_max_pixel update:{:.4f}'.format(epoch_post + 1, epochs, ap_raw_pixel))
                ema.save_check()
            else:
                if patience == early_stop_patience:
                    logger.info("--------------------------------------------------------------------------")
                    logger.info("Start the second stage")
                    logger.info("--------------------------------------------------------------------------")
                    ema.load_check()
                    stage = 2
                    optimizer = optimizer_stage2
                patience = patience + 1
        else:
            if ap_raw_image >= ap_max_image:
                patience = 0
                ap_max_image = ap_raw_image
                logger.info('epoch post [{}/{}], ap_max_image update:{:.4f}'.format(epoch_post + 1, epochs, ap_raw_image))
                ema.save_check()
            else:
                if patience == early_stop_patience:
                    logger.info("early stop")
                    break 
                patience = patience + 1
        


if __name__ == '__main__':

    # Due to our reimplementation of the original code to support more backbones and achieve better zero-shot anomaly detection (ZSAD) performance, 
    # some training parameters differ slightly from those reported in the original paper.
    parser = argparse.ArgumentParser("Bayes-PFL", add_help=True)
    
    # Model
    parser.add_argument("--dataset", type=str, default='visa', help="Training dataset name")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="Source of pretrained weight")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    # path
    parser.add_argument("--train_data_path", type=str, default="./dataset/mvisa/data", help="Training dataset path")
    parser.add_argument("--val_data_path", type=str, default="./dataset/mvisa/data", help="Validation dataset path")
    parser.add_argument("--save_path", type=str, default='./my_exps/train_visa', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip_local/model_configs/ViT-L-14-336.json', help="model configs")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_weight/ViT-L-14-336px.pt", help="Original pretrained CLIP path")

    # hyper-parameter
    parser.add_argument('-nf', '--num_flows', type=int, default=10,
                        metavar='NUM_FLOWS', help='Flow length')  # $K$ in the main text
    parser.add_argument("--prompt_context_len", type=int, default=5, help="The length of learnable context vectors")  # $P$ in the main text
    parser.add_argument("--prompt_num", type=int, default=3, help="The number of prompts in the prompt bank") # $B$ in the main text
    parser.add_argument("--prompt_state_len", type=int, default=5, help="The length of learnable state vectors")  # $Q$ in the main text
    parser.add_argument("--learning_rate", type=float, default= 0.0001, help="learning rate")
    parser.add_argument("--learning_rate_PFL", type=float, default=0.001, help="learning rate")
    parser.add_argument("--learning_rate_prompt", type=float, default=0.001, help="learning rate")
    parser.add_argument("--learning_rate_linear", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--sample_num", type= int, default= 5, help="Number of Monte Carlo samples during validation. Note that during training, we sample only once by default for efficiency")

    # Training
    parser.add_argument("--batch_size", type=int, default= 32, help="batch size")
    parser.add_argument("--image_size", type=int, default= 518, help="image size")
    parser.add_argument("--epochs", type=int, default= 30, help="Maximum training epochs")
    parser.add_argument("--alpha", type= float, default= 0.999, help="Not used in this paper")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--device_id", type=int, default= 0, help="GPU ID")
    parser.add_argument("--seed", type=int, default= 111, help="save frequency")

    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)

    setup_seed(args.seed)
    train(args)