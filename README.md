# Neural Optimal Transport for 3D Mnist samples

The whole repo is based on beautiful [NOT paper](https://openreview.net/forum?id=d8CBRlWNkqH) & corresponding [Repo](https://github.com/iamalexkorotin/NeuralOptimalTransport), with modification such that NOT finds weak optimal transport plan between [3D Mnist samples](https://openreview.net/forum?id=d8CBRlWNkqH).

## Table of Contents
- [NOT in 3D](#Idea)
- [Repository structure](#repo)
- [Examples](#examples)
- [Installation](#install)
- [Running commands](#running)
## NOT in 3D
For base models, we took original [Unet](https://arxiv.org/abs/1505.04597) model and changed all `Conv` layers to `Conv3D`.
## Repository structure
All python scripts are contained in `src` folder. Jupyter notebooks used for small experiments and research are located in `research` folder. For training and see the resutls in cmd interface checkout `run.py` script

## Running commands
To run training script:
```
python run.py
```
Checkpoints and visulization of X and T(X) will be save in ```checkpoints``` directory
## Examples
3D MNSIT Optimal transport from 2 to 4:

X
![X](images/2_24.jpeg)

T(X)
![T(x)](images/4_24.jpeg)


3D MNSIT Optimal transport from 4 to 2:

X
![X_4](images/4_42.jpeg)

T(X)
![T(X)_2](images/2_42.jpeg)


## Installation (Using conda)
To install all required dependencies, you can run
```
conda create --name 3DNOT --file requirements.txt
```

## Credits
Final project for Machine Learning course 2023 at Skoltech was done by Maksim Bobrin, Sergei Kholkin, Anastasia Batsheva, Artem Basharin.