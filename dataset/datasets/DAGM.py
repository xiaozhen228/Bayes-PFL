import numpy as np 
import os 
import shutil 
import cv2
from pycocotools.coco import COCO
import glob
import sys
def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)

class DAGM_dataset():
    def __init__(self,path_root):
        #self.is_binary = True   
        self.is_255 = True  
        self.sub_class = True
        self.path_root = path_root
        if self.sub_class:
            self.dataset_name =  [f"classes{i}" for i in [1,2,3,4,5,7,8]]
            self.dataset_map = dict(zip(self.dataset_name, [f"fabric{i}" for i in [1,2,3,4,5,7,8]]))
        else:
            self.dataset_name =  ["fabric"]
        self.dataset = "DAGM"

        
    def Binary(self,mask, raw_mask_path):
        if self.is_255:
            try:
                assert (np.unique(mask) == np.array([0,255])).all(), f"{raw_mask_path}"
            except AttributeError:
                print(np.unique(mask), raw_mask_path)
                sys.exit(1)
            mask[mask<=128] = 0
            mask[mask>128] = 1
        else:
            mask[mask<=0] = 0
            mask[mask>0] = 1
        return mask
    #.capitalize()   
    def make_dirs(self,des_root):
        for data_name in self.dataset_name:
            name = self.Change_name(data_name)
            dir_list = [os.path.join(des_root,name,"train","good"),os.path.join(des_root,name,"test","good"),os.path.join(des_root,name,"test","anomaly"),os.path.join(des_root,name,"ground_truth","anomaly")]
            for dir in dir_list:
                if not os.path.exists(dir):
                    os.makedirs(dir)

    def Change_name(self,name):
        if self.sub_class:
            name = self.dataset_map[name]
        name_new = name.capitalize()
        return name_new
    

    # train  test  ground_truth
    def make_VAND(self,binary,to_255,des_path_root,id):
        self.make_dirs(des_path_root)
        for data_name in self.dataset_name:
            new_name = self.Change_name(data_name)
            print("Processing:{}".format(data_name))
            anomaly_num = len(glob.glob(os.path.join(self.path_root, "pos",f"*{data_name}*.jpg")))
            temp = 0
            for mode in ["pos","neg"]:
                data_path = os.path.join(self.path_root, mode)
                data_list = glob.glob(os.path.join(data_path,f"*{data_name}*.jpg"))
                data_list = sorted(data_list)
                for i in range(len(data_list)):
                    pic_name = os.path.basename(data_list[i]).split(".")[0]
                    raw_mask_path = os.path.join(data_path,os.path.basename(data_list[i]).replace("jpg","png"))
                    assert os.path.exists(data_list[i]) and os.path.exists(raw_mask_path)
                    raw_img = cv2.imread(data_list[i],cv2.IMREAD_COLOR)
                    raw_mask = cv2.imread(raw_mask_path,cv2.IMREAD_GRAYSCALE)
                    is_good = (np.sum(raw_mask)==0) 
                    if mode == "pos":
                        assert is_good == False, f"{raw_mask_path}"
                    else:
                        assert is_good == True, f"{raw_mask_path}"
                    if binary and mode == "pos":
                        raw_mask = self.Binary(raw_mask.copy(),raw_mask_path)
                        if to_255: 
                            raw_mask =  raw_mask * 255
                    
                    if is_good:
                        if temp == anomaly_num:
                            continue
                        save_img_path = os.path.join(des_path_root,new_name,"test","good",f"{pic_name}_"+str(id).zfill(10)+".bmp")  
                        cv2.imwrite(save_img_path, raw_img)
                        temp = temp + 1

                    else:
                        save_img_path = os.path.join(des_path_root,new_name,"test","anomaly",f"{pic_name}_"+str(id).zfill(10)+".bmp")
                        save_mask_path = os.path.join(des_path_root,new_name,"ground_truth","anomaly",f"{pic_name}_"+str(id).zfill(10)+".png")

                        cv2.imwrite(save_img_path,raw_img)
                        cv2.imwrite(save_mask_path,raw_mask,[int(cv2.IMWRITE_PNG_COMPRESSION),10])  #[int(cv2.IMWRITE_JPEG_QUALITY),100]
                    id = id + 1

        print(f"{self.dataset} finished !")
        return id


