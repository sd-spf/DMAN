import os
import torch
import shutil
import pandas as pd
from .optimizers import *
from config import configs
from torch import optim as optim_t
from tqdm import tqdm
from glob import glob
from itertools import chain
import json

def get_optimizer(model):
    if configs.optim == "adam":
        return optim_t.Adam(model.parameters(),
                            configs.lr,
                            betas=(configs.beta1,configs.beta2),
                            weight_decay=configs.wd)
    elif configs.optim == "radam":
        return RAdam(model.parameters(),
                    configs.lr,
                    betas=(configs.beta1,configs.beta2),
                    weight_decay=configs.wd)
    elif configs.optim == "ranger":
        return Ranger(model.parameters(),
                      lr = configs.lr,
                      betas=(configs.beta1,configs.beta2),
                      weight_decay=configs.wd)
    elif configs.optim == "over9000":
        return Over9000(model.parameters(),
                        lr = configs.lr,
                        betas=(configs.beta1,configs.beta2),
                        weight_decay=configs.wd)
    elif configs.optim == "ralamb":
        return Ralamb(model.parameters(),
                      lr = configs.lr,
                      betas=(configs.beta1,configs.beta2),
                      weight_decay=configs.wd)
    elif configs.optim == "sgd":
        return optim_t.SGD(model.parameters(),
                        lr = configs.lr,
                        momentum=configs.mom,
                        weight_decay=configs.wd)
    else:
        print("%s  optimizer will be add later"%configs.optim)

def save_checkpoint(state,best_acc,k):
    filename = configs.checkpoints + os.sep + configs.model_name + "-"+str(best_acc)+'-'+str(k)+"-checkpoint.pth.tar"
    message = filename.replace("-checkpoint.pth.tar","-best_model.pth.tar")
    torch.save(state, message)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_files(root,mode):
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    else:
        img_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
        # 类别名称排序
        img_class.sort()
        # 生成类别名称以及对应的数字索引
        class_indices = dict((k, v) for v, k in enumerate(img_class))
        # 以{数字索引: 类别名称}的格式保存为json文件，用于将预测的分类索引转换成对应的类别名称
        json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        all_data_path32, all_data_path64,labels = [], [],[]
        image_folders32 = list(map(lambda x: root + x+'/32', os.listdir(root)))
        image_folders64 = list(map(lambda x: root + x + '/64', os.listdir(root)))
        # s= os.path.join(image_folders,'32')
        all_images32 = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders32))))
        all_images64 = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders64))))
        # img_path = os.listdir()
        print("loading train dataset")
        for file in tqdm(all_images32):

            all_data_path32.append(file)

        for file64 in tqdm(all_images64):
            all_data_path64.append(file64)
            clas = class_indices[file64.split('/')[-3]]
            labels.append(clas)

        all_files = pd.DataFrame({"filename_32": all_data_path32,"filename_64":all_data_path64, "label": labels})


        return all_files
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lrs = [5e-4, 1e-4, 1e-5, 1e-6]
    if epoch<=10:
        lr = lrs[0]
    elif epoch>10 and epoch<=16:
        lr = lrs[1]
    elif epoch>16 and epoch<=22:
        lr = lrs[2]
    else:
        lr = lrs[-1]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr