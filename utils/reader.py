from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
class WeatherDataset(Dataset):
    # define dataset
    def __init__(self,label_list,transforms32=None,transforms64=None,mode="train"):
        super(WeatherDataset,self).__init__()
        self.label_list = label_list
        self.transforms32 = transforms32
        self.transforms64 = transforms64
        self.mode = mode
        imgs = []
        if self.mode == "test":
            for index,row in label_list.iterrows():
                imgs.append((row["filename_32"],row["filename_64"]))
            self.imgs = imgs
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename_32"],row["filename_64"],row["label"]))
            self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        if self.mode == "test":
            filename = self.imgs[index]
            img = Image.open(filename).convert('RGB')
            img = self.transforms(img)
            return img,filename
        else:
            filename32,filename64,label = self.imgs[index]
            img32 = Image.open(filename32).convert('RGB')
            img32 = self.transforms32(img32)

            img64 = Image.open(filename64).convert('RGB')
            img64 = np.array(img64)
            img64 = cv2.cvtColor(img64, cv2.COLOR_RGB2Lab)
            img2_l, img2_a, img2_b = np.split(img64, 3, axis=2)
            img64 = self.transforms64(img2_l)
            return img32,img64,label


