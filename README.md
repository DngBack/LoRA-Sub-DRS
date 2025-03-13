# LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning

<div align="justify">
  This repository contains the official implementation of our CVPR 2025 paper, "LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning."
</div>

## Requisite

This code is implemented in PyTorch, and we perform the experiments under the following environment settings:

- python = 3.11.4
- torch = 2.0.1
- torchvision = 0.15.2
- timm = 0.6.7

The code has been tested on Linux Platform with a GPU (RTX3080 Ti).


## Dataset 
 * Create a folder `data/`
 * **CIFAR 100**: should automatically be downloaded
 * **ImageNet-R**: retrieve from [link](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar). After unzipping, place it into `data/` folder 
 * **CUB200**: retrieve from [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing), place it into `data/` folder
 * **DomainNet**: retrieve from [link](http://ai.bu.edu/M3SDA/), place it into `data/` folder

## Training
You can modify "init_cls" and "increment" parameters in `configs/[dataset].json` to configure different CIL settings.

- CIFAR100:
    ```
    python main.py --device your_device --config configs/cifar100.json 
    ```

- ImageNet-R:
    ```
    python main.py --device your_device --config configs/imagenetr.json 
    ```
  
- CUB (20 Task):
    ```
    python main.py --device your_device --config configs/cub.json 
    ```

- DomainNet (5 Task):
    ```
    python main.py --device your_device --config configs/domainnet.json 
    ```


## Citation

```bibtex

```


## Reference
We appreciate the following repositories for their contributions of useful components and functions to our work.

- [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt)
- [InfLoRA](https://github.com/liangyanshuo/InfLoRA)



