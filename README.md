

# DMAN
“An AI-assisted fluorescence microscopic system for screening mitophagy inducers by simultaneous analysis of mitophagic intermediates” in AI Framework-DMAN

# Prerequisite

> - CUDA/CUDNN
> - Python3
> - PyTorch==1.13
```
3. Prepare dataset

Organize the folder as follows:
```
1. 这是一级的有序列表，数字1还是1
   1. 这是二级的有序列表，阿拉伯数字在显示的时候变成了罗马数字
      1. 这是三级的有序列表，数字在显示的时候变成了英文字母
         
├── ../../dataset/
  ├── train/     
    ├── class1/
      ├── 32/
      ├── 128/
    ├── class2/
      ├── 32/
      ├── 128/
    ...
  ├── test/
    ├── images1.png
    ├── images2.png
...
```
# Training and Evaluation example

> Training and evaluation are on a single GPU.

### Train with unsupervised domain adaptation 

```
python main.py
```
### Evaluation

```
python test.py
```


