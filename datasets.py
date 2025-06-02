import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
import albumentations as A
import torchvision.transforms as transforms 
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
import cv2


class Makedataset():
	def __init__(self, train_data_path, preprocess_test, mode, image_size = 518):
		self.train_data_path= train_data_path
		self.preprocess_test = preprocess_test
		self.mode = mode
		self.target_transform = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.CenterCrop(image_size),transforms.ToTensor()])

	def mask_dataset(self, name, product_list, batchsize, shuf = True):
		dataset = MyDataset(root=self.train_data_path, transform=self.preprocess_test, target_transform=self.target_transform, 
											mode =self.mode , product_list= product_list, dataset = name)
		obj_list = dataset.get_cls_names()
		
		dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = shuf)
		
		return dataloader, obj_list
		
class MyDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='train', dataset = "visa", product_list = None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.mode = mode
		self.dataset = dataset
		self.aug = True
		self.aug_rate = 0.4

		if dataset in ["HeadCT", "BrainMRI", "Br35H"]:
			self.no_gt = True 
		else:
			self.no_gt = False
		self.generate_anomaly = False
		print(f"self.generate_anomaly: {self.generate_anomaly}")
		anomaly_source_path = "path/dtd/dtd/images"
		#self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

		self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

		self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

		self.resize_shape = (1024,1024)

		if "test" in mode:
			self.aug = False
		self.data_all = []

		if mode == "train":
			meta_info = json.load(open(f'{self.root}/meta_{self.dataset}.json', 'r'))
			meta_info = meta_info["test"]
			if product_list is not None:
				keys = meta_info.keys()
				keys = list(keys)
				for key in keys:
					if key not in product_list:
						del meta_info[key]

			self.cls_names = list(meta_info.keys())
			#self.cls_names = ["chewinggum"]
			for cls_name in self.cls_names:
				self.data_all.extend(meta_info[cls_name])

		elif mode == "test":
			meta_info = json.load(open(f'{self.root}/meta_{self.dataset}.json', 'r'))
			meta_info = meta_info["test"]
			if product_list is not None:
				keys = meta_info.keys()
				keys = list(keys)
				for key in keys:
					if key not in product_list:
						del meta_info[key]

			self.cls_names = list(meta_info.keys())
			#self.cls_names = ["bottle"]
			for cls_name in self.cls_names:
				self.data_all.extend(meta_info[cls_name])
		
		self.length = len(self.data_all)
		self.img_trans = A.Compose([
			A.Rotate(limit=30, p=0.5),
			A.RandomRotate90( p = 0.5),
			A.RandomBrightnessContrast(p=0.5),
			A.GaussNoise(p=0.5),
			A.OneOf([
				A.Blur(blur_limit=3, p=0.5),
				A.ColorJitter(p=0.5),
				A.GaussianBlur(p=0.5),
			], p=0.5)
		], is_check_shapes=False)

	def randAugmenter(self):
		aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
		aug = iaa.Sequential([self.augmenters[aug_ind[0]],
								self.augmenters[aug_ind[1]],
								self.augmenters[aug_ind[2]]]
								)
		return aug

	def augment_image(self, image, mask, anomaly_source_path): # The synthetic anomaly strategy from DRAEM, which has not been applied in our work, can be ignored.
		
		image = np.array(image, dtype= np.uint8)[:, :, ::-1]
		mask = np.array(mask, dtype= np.uint8)
		image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[1]), interpolation=cv2.INTER_CUBIC)
		image = image / 255.0
		mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[1]),interpolation=cv2.INTER_NEAREST )
		h, w = image.shape[0], image.shape[1]

		aug = self.randAugmenter()
		perlin_scale = 3
		min_perlin_scale = 1
		anomaly_source_img = cv2.imread(anomaly_source_path)
		anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(w, h))

		anomaly_img_augmented = aug(image=anomaly_source_img)
		perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
		perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

		perlin_noise = rand_perlin_2d_np((h, w), (perlin_scalex, perlin_scaley))
		perlin_noise = self.rot(image=perlin_noise)
		threshold = 0.5
		perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
		perlin_thr = np.expand_dims(perlin_thr, axis=2)

		img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

		beta = torch.rand(1).numpy()[0] * 0.8

		augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
			perlin_thr)

		no_anomaly = torch.rand(1).numpy()[0]
		if no_anomaly > 1:
			image = np.array(image*255, dtype= np.uint8)[:, :, ::-1]
			msk = np.array(mask, dtype= np.uint8)* 255
			image = Image.fromarray(image)
			msk = Image.fromarray(msk, mode = "L")
			return image, msk , False
		else:
			#print("---------------")
			augmented_image = augmented_image.astype(np.float32)
			msk = (perlin_thr).astype(np.float32)
			augmented_image = msk * augmented_image + (1-msk)*image

			msk = np.squeeze(msk) + mask
			msk[msk!=0] = 1.0
			has_anomaly = 1
			if np.sum(msk) == 0:
				has_anomaly= 0
			
			augmented_image = np.array(augmented_image*255, dtype= np.uint8)[:, :, ::-1]
			msk = np.array(msk, dtype= np.uint8) * 255
			augmented_image = Image.fromarray(augmented_image)
			msk = Image.fromarray(msk, mode = "L")
			#return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)
			return augmented_image, msk, has_anomaly



	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def MyCrop(self, img, mask, CropSize):

		if np.sum(mask)  == 0:
			return img, mask
		#print("crop")
		h, w = img.shape[:2]
		mask_indices = np.where(mask > 0)
		x_min = np.min(mask_indices[1])
		y_min = np.min(mask_indices[0])
		x_max = np.max(mask_indices[1])
		y_max = np.max(mask_indices[0])

		if CropSize[1] < w :
			crop_col = random.randint(max(0, x_min - CropSize[1]), min(x_max, w - CropSize[1]))
		else:
			crop_col = 0
		
		if CropSize[0] < h:
			crop_row = random.randint(max(0, y_min - CropSize[0]), min(y_max, h - CropSize[0]))
		else:
			crop_row = 0

		crop_width = min(CropSize[1], w - crop_col)
		crop_height = min(CropSize[0], h - crop_row)

		img = img[crop_row:crop_row + crop_height, crop_col:crop_col + crop_width]
		mask = mask[crop_row:crop_row + crop_height, crop_col:crop_col + crop_width]
		return img, mask


	def combine_img(self, cls_name):   # From APRIL-GAN: https://github.com/ByChelsea/VAND-APRIL-GAN
		img_paths = os.path.join(self.root, "mvtec",cls_name, 'test')
		img_ls = []
		mask_ls = []
		for i in range(4):
			defect = os.listdir(img_paths)
			#random_defect = random.choice(defect)
			random_defect = "anomaly"
			files = os.listdir(os.path.join(img_paths, random_defect))
			random_file = random.choice(files)
			img_path = os.path.join(img_paths, random_defect, random_file)
			mask_path = os.path.join(self.root, "mvtec",cls_name, 'ground_truth', random_defect, random_file.replace("bmp", "png"))
			assert (os.path.exists(img_path))
			img = Image.open(img_path)
			img_ls.append(img)
			if random_defect == 'good':
				img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
			else:
				assert os.path.exists(mask_path)
				img_mask = np.array(Image.open(mask_path).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
			mask_ls.append(img_mask)
		# image
		image_width, image_height = img_ls[0].size
		result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
		for i, img in enumerate(img_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_image.paste(img, (x, y))

		# mask
		result_mask = Image.new("L", (2 * image_width, 2 * image_height))
		for i, img in enumerate(mask_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_mask.paste(img, (x, y))

		return result_image, result_mask


	def Trans( self, img , img_mask):
		corp_list = [(512,512), (600, 600), (768, 768), (700, 700), (800,800), (896, 896), (1024, 1024), (1280, 1280)]
		img_mask = np.array(img_mask)
		img = np.array(img)[:, :, ::-1]

		#idex_crop = np.random.randint(0, len(corp_list), 1)
		#pro_crop = np.random.rand()
		#if pro_crop < 0.3:
		#	img, img_mask = self.MyCrop(img, img_mask, corp_list[idex_crop[0]])

		augmentations = self.img_trans(mask=img_mask, image=img)
		img = augmentations["image"][:, :, ::-1]
		img_mask = augmentations["mask"]
		img = Image.fromarray(img.astype(np.uint8))
		img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
		return img, img_mask


	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															data['specie_name'], data['anomaly']
		random_number = random.random()
		#random_number = 0
		if random_number < self.aug_rate and self.dataset == "mvtec" and self.mode == "train":
			img, img_mask = self.combine_img(cls_name)   
			anomaly = 1
		else:
			img = Image.open(os.path.join(self.root, img_path))
			if anomaly == 0:
				img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
			else:
				if self.no_gt:
					img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
				else:
					img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
					img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

		if self.mode == "train" and self.generate_anomaly == True:
			random_number = random.random()
			if random_number > 0.5:
				anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
				img, img_mask, anomaly = self.augment_image(img,img_mask, self.anomaly_source_paths[anomaly_source_idx])
			else:
				img, img_mask = self.Trans(img, img_mask)

		
		if self.mode == "train" and self.aug == True:
			img, img_mask = self.Trans(img, img_mask)
		

		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}
	