import os
import json

import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms.functional as F
from models.model import resnet34
from torchvision import transforms, datasets, utils
from tqdm import tqdm
import sys
def custom_colormap(im,gray_image):
    # 自定义调色板
    custom_palette = np.array([[0, 0, 0],    # 红色 (0)
                               [255, 0, 0],    # 绿色 (0.25)
                               [0, 0, 255],    # 蓝色 (0.75)
                               [0, 255, 0]], dtype=np.uint8)   # 黄色 (1)

    # 创建一个灰度到伪彩色的映射表
    h, w = gray_image.shape
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            idx = int(gray_image[i, j])  # 将0~1范围的值映射到0~3.99，并转换为整数索引
            colored_image[i, j] = custom_palette[idx]
    im3 = cv2.addWeighted(im, 1, colored_image, 0.3, 0)
    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_RGB2BGR)
    return im3,colored_image

if __name__=='__main__':
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    data_transformRGB = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transformL = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize (mean=[0.456],std=[0.224])])

    # load image

    img_path = "test_images/"
    out_dir = "./result/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    L= os.listdir(img_path)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = resnet34(channel=3,num_classes=3).to(device)

    best_cpk = "./checkpoints/best_model.pth.tar" # load model weights path
    checkpoint = torch.load(best_cpk)
    window_size = 32

    file = open('result.txt', 'w')

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    RGBL_padding_size = 128-window_size

    stride = 1
    for k in L:
        print("name:{}".format(k))
        save_file1024 = os.path.join(out_dir,k)

        img1=img_path+'/'+k

        img_1024 = cv2.imread(img1)
        img_1024[:, :, 2] = 0
        H, W, C = img_1024.shape
        image_area = H * W

        mat_1024  = np.zeros((H,W))
        # 计算每个小块的大小
        a, b, l, m = 0, 0, 0, 0
        # 切割图像并保存每个小块
        for i in range(0, H - window_size + 1, stride):
            for j in range(0, W - window_size + 1, stride):

                img_bgr = img_1024[i:window_size + i, j:window_size + j,:]

                if window_size+i+RGBL_padding_size >=H or window_size+j+RGBL_padding_size >=W :
                    if window_size+j+RGBL_padding_size>=W and window_size+i+RGBL_padding_size <H:
                        if window_size+j+RGBL_padding_size==W:
                            e=j
                        img_lab = img_1024[i:window_size + i+RGBL_padding_size, e:W, :]
                    if window_size+i+RGBL_padding_size>=H and window_size+j+RGBL_padding_size<W:
                        if window_size + i + RGBL_padding_size == H:
                            d = i
                        img_lab = img_1024[d:H, j:window_size + j+RGBL_padding_size, :]
                    if window_size+i+RGBL_padding_size>=H and window_size+j+RGBL_padding_size>=W:
                        if window_size + j + RGBL_padding_size == W:
                            e = j
                        if window_size + i + RGBL_padding_size == H:
                            d=i
                        img_lab = img_1024[d:H, e:W,:]
                else:
                    img_lab = img_1024[i:window_size + i+RGBL_padding_size, j:window_size + j+RGBL_padding_size, :]

                gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                black_pixel_count = np.sum(gray_image < 18)

                total_pixels = window_size * window_size
                black_ratio = black_pixel_count / total_pixels
                print(black_ratio)
                if black_ratio>0.9:
                    if (window_size+i == H and window_size + j == W):
                        b=b+window_size*window_size
                    elif window_size+j == W or window_size+i == H:
                        b=b+window_size
                    else:
                        b=b+1
                else:
                    area = np.argwhere(gray_image > 5)
                    backarea = np.argwhere(gray_image <= 5)
                    img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
                    img_lab = cv2.cvtColor(img_lab,cv2.COLOR_RGB2Lab)

                    img2_l, img2_a, img2_b = np.split(img_lab, 3, axis=2)

                    imgRGBL = data_transformL(img2_l)
                    imgRGB = data_transformRGB(img_rgb)

                    imgRGBL = torch.unsqueeze(imgRGBL, dim=0)
                    imgRGB = torch.unsqueeze(imgRGB, dim=0)
                    with torch.no_grad():

                        output = model(imgRGB.to(device),imgRGBL.to(device))

                        output = torch.squeeze(output).cpu()

                        predict = torch.softmax(output, dim=0)

                        predict_cla = torch.argmax(predict).numpy()

                        print_res_block = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                                                   predict[predict_cla].numpy())
                        print(print_res_block)
                        if (window_size + j == W and window_size+i == H):

                            if class_indict[str(predict_cla)] == 'autophagy':
                                c = float(3)
                                a = a + area.shape[0]
                                b = b + backarea.shape[0]
                            elif class_indict[str(predict_cla)] == 'mitochondrion':
                                c = float(2)
                                m = m + area.shape[0]
                                b = b + backarea.shape[0]
                            else:
                                c = float(1)
                                l = l + area.shape[0]
                                b = b + backarea.shape[0]

                        elif window_size+j == W or window_size+i == H :

                            if class_indict[str(predict_cla)] == 'autophagy':
                                c = float(3)
                                a = a+window_size
                            elif class_indict[str(predict_cla)] == 'mitochondrion':
                                c = float(2)
                                m = m + window_size
                            else:
                                c = float(1)
                                l= l+ window_size
                        else:
                            if class_indict[str(predict_cla)] == 'autophagy':
                                c = float(3)
                                a = a + 1
                            elif class_indict[str(predict_cla)] == 'mitochondrion':
                                c = float(2)
                                m = m + 1
                            else:
                                c = float(1)
                                l = l + 1
                        for idx in area:
                            mat_1024[i+idx[0], j+ idx[1]] = c

        img_1024 = cv2.cvtColor(img_1024,cv2.COLOR_BGR2RGB)
        heat_map,color = custom_colormap(img_1024, mat_1024)
        cv2.imwrite(save_file1024, heat_map)
        cv2.imwrite(save_file1024[:-4] + '_color.png', color)
        file.write(str("name:{}".format(k)+ '\n'))
        file.write(str("class: autophagy  proportion: {},{}".format(a/ image_area,a)+ '\n'))
        file.write(str("class: background  proportion: {},{}".format(b/ image_area,b)+ '\n'))
        file.write(str("class: lysosome  proportion:{},{}".format(l/ image_area,l)+ '\n'))
        file.write(str("class: mitochondrion  proportion: {},{}".format(m/ image_area,m)+ '\n'))
    file.close()

