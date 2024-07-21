

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

├── ../../dataset/
│   ├── BraTs/     
|   |   ├── images/
|   |   ├── labels/
|   |   ├── T1/
|   |   ├── T1c/
|   |   ├── T2/
│   ├── FeTS15/
|   |   ├── Flair/
|   |   ├── labels/
|   |   ├── T1/
|   |   ├── T1c/
|   |   ├── T2/
│   ├── xinhua/ 
|   |   ├── Flair/
|   |   ├── labels/
|   |   ├── T1/
|   |   ├── T1c/
|   |   ├── T2/

├── ../../dataset/
│   ├── train/     
|   |   ├── class1/
|   |   |   ├── 32/
|   |   |   ├── 128/
|   |   ├── class2/
|   |   |   ├── 32/
|   |   |   ├── 128/
│   ├── test/
|   |   ├── images1.png
|   |   ├── images2.png
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


