import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from models.VPB import Context_Prompting
from models.metric_and_visualization import calcuate_metric_pixel, calcuate_metric_image
import open_clip_local as open_clip
from models.VPB import TextEncoder
from models.model_CLIP import Load_CLIP, tokenize
from datasets import Makedataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
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


def test(args):
    image_size = args.image_size
    save_path = args.save_path
    dataset_name = args.dataset

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    txt_path = os.path.join(save_path, 'log.txt')

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

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    PFL_TextEncoder = TextEncoder(model_clip, args)
    MyModel = Context_Prompting(args = args).to(device)
    MyModel.eval()

    checkpoint = torch.load(args.checkpoint_path, map_location= device)
    MyModel.load_state_dict(checkpoint["MyModel"], strict= True)

    preprocess_test = _transform_test(image_size)


    Make_dataset_test = Makedataset(train_data_path = args.data_path, preprocess_test = preprocess_test, mode = "test", 
                               image_size = args.image_size)
    test_dataloader, obj_list = Make_dataset_test.mask_dataset(name = dataset_name, product_list= None, batchsize = 1, shuf= False)

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []

    results['pr_sp'] = []
    results['gt_sp'] = []
    results['path'] = []

    id = 0
    if "epoch_post" in args.checkpoint_path:
        stage = int(args.checkpoint_path[:-4].split("_")[-1])
    else:
        stage = 2
    for items in tqdm(test_dataloader):
        id = id + 1
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        results['gt_sp'].append(items['anomaly'].item())
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        
        with torch.no_grad():
            image_features, _ , patch_tokens =  model_clip.encode_image(image, args.features_list)
            text_embeddings, _ = MyModel.forward_ensemble(PFL_TextEncoder, image_features, patch_tokens ,cls_name, device, tokenizer, mode = "test") # # B * R text embeddings
            temp_cls = 0
            pro_img, anomaly_maps_list = MyModel(text_embeddings, image_features, patch_tokens, stage = stage, mode = "test")
            pro_img = pro_img.squeeze(2)
            for i in range(args.prompt_num * args.sample_num): # B * R anomaly scores
                text_probs = torch.cat([pro_img[:,i].unsqueeze(0), pro_img[:,i + args.prompt_num * args.sample_num].unsqueeze(0)], dim = 1).softmax(dim = -1)
                temp_cls = temp_cls + text_probs[0, 1]
            temp_cls = temp_cls / (args.prompt_num * args.sample_num)

            anomaly_maps = []
            for num in range(len(anomaly_maps_list)):
                anomaly_map = anomaly_maps_list[num]
                for i in range(args.prompt_num * args.sample_num):  # B * R anomaly maps
                    temp = torch.softmax(torch.stack([anomaly_map[:,i,:,:], anomaly_map[:,i+(args.prompt_num * args.sample_num) ,:,:]], dim = 1), dim =1)
                    anomaly_maps.append(temp[:, 1, :, :].cpu().numpy())
            anomaly_map = np.mean(anomaly_maps, axis=0)[0]
            results['anomaly_maps'].append(anomaly_map)
            results['pr_sp'].append(temp_cls.cpu().numpy())
        path = items['img_path']
        results['path'].extend(path)

    datasets_only_classification =  ["HeadCT", "BrainMRI", "Br35H"]  # These datasets lack ground truth, so only zero-shot anomaly classification metrics are calculated.
    if args.dataset in datasets_only_classification:
        calcuate_metric_image(results, obj_list, logger, alpha = 0.5 , sigm = 8, args = args)
    else:
        calcuate_metric_pixel(results, obj_list, logger, alpha = 0.5 , sigm = 8, args = args)

import shutil
def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
if __name__ == '__main__':

    # Industry_Datasets = ["mvtec","visa", "BTAD", "RSDD", "KSDD2", "DAGM", "DTD"]
    # Medical_Datasets = ["HeadCT", "BrainMRI", "Br35H", "ISIC", "ClinicDB", "ColonDB", "Kvasir", "Endo"]

    # datasets_no_good = ["ISIC", "ClinicDB", "ColonDB", "Kvasir", "Endo"] # These datasets do not contain normal samples, so only zero-shot anomaly segmentation metrics are calculated.
    # datasets_only_classification = ["HeadCT", "BrainMRI", "Br35H"]  # These datasets lack ground truth, so only zero-shot anomaly classification metrics are calculated.

    parser = argparse.ArgumentParser("Bayes-PFL", add_help=True)

    # Model
    parser.add_argument("--dataset", type=str, default='mvtec', help="Testing dataset name")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--image_size", type=int, default= 518, help="image size")
    parser.add_argument("--pretrained", type=str, default="openai", help="Source of pretrained weight")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    # path
    parser.add_argument("--data_path", type=str, default="./dataset/mvisa/data", help="Testing dataset path")
    parser.add_argument("--save_path", type=str, default='./results/test_mvtec', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip_local/model_configs/ViT-L-14-336.json', help="model configs")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_weight/ViT-L-14-336px.pt", help="Original pretrained CLIP path")
    parser.add_argument("--checkpoint_path", type=str, default="./bayes_weight/train_visa.pth", help='path to checkpoint')

    # hyper-parameter
    parser.add_argument('-nf', '--num_flows', type=int, default=10,
                        metavar='NUM_FLOWS', help='Flow length')  # $K$ in the main text
    parser.add_argument("--prompt_context_len", type=int, default=5, help="The length of learnable context vectors")  # $P$ in the main text
    parser.add_argument("--prompt_num", type=int, default=3, help="The number of prompts in the prompt bank") # $B$ in the main text
    parser.add_argument("--prompt_state_len", type=int, default=5, help="The length of learnable state vectors")  # $Q$ in the main text
    parser.add_argument("--sample_num", type= int, default= 10, help="The number of Monte Carlo sampling interations")   # $R$ in the main text


    parser.add_argument("--device_id", type=int, default= 2, help="GPU ID")
    parser.add_argument("--seed", type=int, default= 111, help="save frequency")


    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)

    if "ceshi" in args.save_path:
        move(args.save_path)
    setup_seed(args.seed)
    test(args)

