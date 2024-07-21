

# DMAN
“An AI-assisted fluorescence microscopic system for screening mitophagy inducers by simultaneous analysis of mitophagic intermediates” in AI Framework-DMAN

# Prerequisite
```
> - CUDA/CUDNN
> - Python3
> - PyTorch==1.13
```
3. Prepare dataset

Organize the folder as follows:

```
|-- dataset/
|   |-- train/
|   |   |-- class1
|   |   |   |-- 32
|   |   |   |-- 128
|   |   |-- class1
|   |   |   |-- 32
|   |   |   |-- 128
         ...
|   |-- test/
|   |-- image1.png
|   |-- image2.png
...
```
# Training and Evaluation example

> Training and evaluation are on a single GPU.

### Train with unsupervised domain adaptation 

```
python main.py
```
### Evaluation
Download our result checkpoint and test sample image from following:[URL](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EawsLUvLsG5LoOeJxYdF5g0BMcv-n6Wn40ETeDDtNyeDmg?e=WMjIqx)
```
python test.py
```


