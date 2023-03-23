# Neural Optimal Transport for 3D Mnist samples

The whole repo is based on beautiful [NOT paper](https://openreview.net/forum?id=d8CBRlWNkqH) & corresponding [Repo](https://github.com/iamalexkorotin/NeuralOptimalTransport), with modification such that NOT finds weak optimal transport plan between [3D Mnist samples](https://openreview.net/forum?id=d8CBRlWNkqH).

## Table of Contents
- [NOT in 3D](#Idea)
- [Repository structure](#repo)
- [Examples](#examples)
- [Installation](#install)
- [Running commands](#running)
## NOT in 3D
----
For base models, we took original [Unet](https://arxiv.org/abs/1505.04597) model and changed all `Conv` layers to `Conv3D`.
## Repository structure
------
All python scripts are contained in `src` folder. Jupyter notebooks used for small experiments and research are located in `research` folder.

## Running commands
----
To run training script 
## Examples
-----

## Installation (Using conda)
To install all required dependencies, you can run
```
conda create --name 3DNOT --file requirements.txt
```