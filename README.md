

# DMAN
“An AI-assisted fluorescence microscopic system for screening mitophagy inducers by simultaneous analysis of mitophagic intermediates” in AI Framework-DMAN

![Model](fig1.tif).

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

```
python test.py
```


