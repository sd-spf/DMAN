

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
Download our result checkpoint and test sample image from following: [URL](https://drive.google.com/drive/folders/1M9d9azwfhCnQ4wwZkUgq1_hRBSgx3JdW?usp=drive_link)
```
python test.py
```


