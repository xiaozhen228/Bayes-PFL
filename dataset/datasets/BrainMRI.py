import numpy as np 
import os 
import shutil 
import cv2
import glob
import random
seed = 228
np.random.seed(seed)
random.seed(seed)

def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


class BrainMRI_dataset():
    def __init__(self,path_root):
        #self.is_binary = True
        self.is_255 = True
        self.path_root = path_root
        
        self.dataset_name = [
        'object'
        ]
        self.dataset_map = {"object": "brain ct"}

    def Binary(self,mask):
        if self.is_255:
            mask[mask<=128] = 0
            mask[mask>128] = 1
        else:
            mask[mask<=0] = 0
            mask[mask>0] = 1
        return mask
    
    def make_dirs(self,des_root):
        for data_name in self.dataset_name:
            name = self.dataset_map[data_name]
            dir_list = [os.path.join(des_root,name,"train","good"),os.path.join(des_root,name,"test","good"),os.path.join(des_root,name,"test","anomaly"),os.path.join(des_root,name,"ground_truth","anomaly")]
            for dir in dir_list:
                if not os.path.exists(dir):
                    os.makedirs(dir)

    # train  test  ground_truth
    def make_VAND(self,binary,to_255,des_path_root,id):
        self.make_dirs(des_path_root)
        for data_name in self.dataset_name:
            print("Processing:{}".format(data_name))
            for mode in ["train","test"]:
                data_path = os.path.join(self.path_root,data_name)
                defect_classes = sorted(os.listdir(os.path.join(data_path,mode)))
                if len(defect_classes) == 1 and "good" in defect_classes and mode=="train":
                    data_list = glob.glob(os.path.join(data_path,mode,"good",'*.jpg'))
                    data_list = sorted(data_list)
                    for i in range(len(data_list)):
                        raw_img = cv2.imread(data_list[i],cv2.IMREAD_COLOR)
                        img_name_id = os.path.basename(data_list[i]).split(".")[0]
                        new_class_name = self.dataset_map[data_name]
                        save_img_path = os.path.join(des_path_root,new_class_name,"train","good","BrainMRI_"+str(img_name_id)+ "_" + str(id).zfill(6)+".bmp")  
                        cv2.imwrite(save_img_path,raw_img)
                        id = id + 1
                else:
                    if len(defect_classes) >= 1 and mode == "test":
                        for classes in defect_classes:
                            if classes == "good":
                                data_list = glob.glob(os.path.join(data_path,mode,classes,'*.jpg'))
                            elif classes == "defect":
                                data_list = glob.glob(os.path.join(data_path,mode,classes,'*.JPG'))
                            data_list = sorted(data_list)
                            if classes == "good":
                                for i in range(len(data_list)):
                                    img_name_id = os.path.basename(data_list[i]).split(".")[0]
                                    raw_img = cv2.imread(data_list[i],cv2.IMREAD_COLOR)
                                    new_class_name = self.dataset_map[data_name]
                                    save_img_path = os.path.join(des_path_root,new_class_name,"test","good","BrainMRI_"+"good_"+str(img_name_id)+ "_" +str(id).zfill(6)+".bmp") 
                                    cv2.imwrite(save_img_path,raw_img)
                                    id = id + 1
                            else:
                                for i in range(len(data_list)):
                                    img_name_id = os.path.basename(data_list[i]).split(".")[0]

                                    raw_img = cv2.imread(data_list[i],cv2.IMREAD_COLOR)

                                    new_class_name = self.dataset_map[data_name]
                                    save_img_path = os.path.join(des_path_root,new_class_name,"test","anomaly",f"BrainMRI_{classes}_"+str(img_name_id)+ "_" +str(id).zfill(6)+".bmp")
                                    cv2.imwrite(save_img_path,raw_img)
                                    id = id + 1
        print("BrainMRI finished !")
        return id
        

                                
                               


