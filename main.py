import random
import sys
import time
import warnings

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import ImageFile
from config import configs
from utils.misc import *
from utils.logger import *
from utils.reader import WeatherDataset
from models.model import resnet34

if configs.fp16:
    try:
        import apex
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
from sklearn.model_selection import StratifiedKFold
# set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(configs.seed)

# make dir for use
def makdir():
    if not os.path.exists(configs.checkpoints):
        os.makedirs(configs.checkpoints)
    if not os.path.exists(configs.log_dir):
        os.makedirs(configs.log_dir)
makdir()

best_acc = 0  # best test accuracy
best_loss = 999 # lower loss

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2,0.5,0.3], gamma=0, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


def main():

    start_epoch = configs.start_epoch
    # set normalize configs for imagenet
    normalize_imgnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    normalize_imgnet64 = transforms.Normalize(mean=[0.456],
                                            std=[0.224])
    transform_train32 = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90, expand=False),
        transforms.ToTensor(),
        normalize_imgnet
    ])
    
    transform_val32 = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize_imgnet
    ])

    transform_train64 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90, expand=False),
        normalize_imgnet64
    ])

    transform_val64 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(128),
        transforms.CenterCrop(128),

        normalize_imgnet64
    ])
    train_files = get_files(configs.dataset,"train")
    k=0
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    X = train_files
    y = train_files['label']

    logger = Logger(os.path.join(configs.log_dir, '%s_log.txt' % configs.model_name), title=configs.model_name)
    logger.set_names(['K', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    for train_index ,val_index in skf.split(X,y):
        k=k+1
        best_acc=0
        best_loss=1000
        print('-----------------第{}折-------------------'.format(k))
        train_data = train_files.iloc[train_index]
        val_data = train_files.iloc[val_index]
        train_dataset = WeatherDataset(train_data, transform_train32, transform_train64)
        val_dataset = WeatherDataset(val_data, transform_val32, transform_val64)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=configs.bs, shuffle=True,
            num_workers=configs.workers, pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=configs.bs, shuffle=False,
            num_workers=configs.workers, pin_memory=True
        )
        model = resnet34(num_classes=configs.num_classes)
        model.cuda()
        criterion = MultiClassFocalLossWithAlpha().cuda()

        optimizer = get_optimizer(model)
        # set lr scheduler method
        if configs.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif configs.lr_scheduler == "on_loss":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5,
                                                                   verbose=False)
        elif configs.lr_scheduler == "on_acc":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5,
                                                                   verbose=False)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
        # for fp16
        if configs.fp16:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level=configs.opt_level,
                                              keep_batchnorm_fp32=None if configs.opt_level == "O1" else configs.keep_batchnorm_fp32
                                              )

    # Train and val
        for epoch in range(start_epoch, configs.epochs):

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, configs.epochs, optimizer.param_groups[0]['lr']))

            train_loss, train_acc, train_5 = train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_acc, test_5 = validate(val_loader, model, criterion, epoch)
            # adjust lr
            if configs.lr_scheduler == "on_loss":
                scheduler.step(val_loss)
            elif configs.lr_scheduler == "on_acc":
                scheduler.step(val_acc)
            elif configs.lr_scheduler == "step":
                scheduler.step(epoch)
            elif configs.lr_scheduler == "adjust":
                adjust_learning_rate(optimizer,epoch)
            else:
                scheduler.step(epoch)
            # append logger file
            lr_current = get_lr(optimizer)
            logger.append([k,lr_current,train_loss, val_loss, train_acc, val_acc])
            print('train_loss:%f, val_loss:%f, train_acc:%f, train_5:%f, val_acc:%f, val_5:%f' % (train_loss, val_loss, train_acc, train_5, val_acc, test_5))

            # save model
            is_best = val_acc > best_acc
            is_best_loss = val_loss < best_loss
            best_acc = max(val_acc, best_acc)
            best_loss = min(val_loss,best_loss)

            save_checkpoint({
                'fold': 0,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'train_acc': train_acc,
                'acc': val_acc,
                'best_acc': best_acc,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            },best_acc,k)
        print('Best acc:')
        print(best_acc)
    logger.close()
def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs32,inputs64, targets) in enumerate(tqdm(train_loader,file=sys.stdout)):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs32, inputs64,targets = inputs32.cuda(), inputs64.cuda(),targets.cuda()
        # print(targets)
        inputs32, inputs64,targets = torch.autograd.Variable(inputs32),torch.autograd.Variable(inputs64), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs32,inputs64)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 3))
        losses.update(loss.item(), inputs32.size(0))
        top1.update(prec1.item(), inputs32.size(0))
        top5.update(prec5.item(), inputs32.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if configs.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (losses.avg, top1.avg, top5.avg)

def validate(val_loader, model, criterion, epoch):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar('Validating: ', max=len(val_loader))
    with torch.no_grad():
        for batch_idx, (inputs32,inputs64, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs32, inputs64, targets = inputs32.cuda(), inputs64.cuda(), targets.cuda()
            inputs32, inputs64, targets = torch.autograd.Variable(inputs32), torch.autograd.Variable(
                inputs64), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs32,inputs64)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 3))
            top1.update(prec1.item(), inputs32.size(0))
            top5.update(prec5.item(), inputs32.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    return (losses.avg, top1.avg, top5.avg)

if __name__ == '__main__':
    main()
