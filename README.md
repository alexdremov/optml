# Gradient Clipping for Coping with Heavy-Tailed Noise in Neural Networks

This repository contains the materials for the semester project "Gradient Clipping for Coping with Heavy-Tailed Noise in Neural Networks." The project investigates the impact of gradient clipping on the training of neural networks, particularly in the context of heavy-tailed noise.

## Authors

- **Vsevolod Skorokhodov**
  - SCIPER: 389703
  - Email: vsevolod.skorokhodov@epfl.ch

- **Andrei Semenov**
  - SCIPER: 388983
  - Email: andrii.semenov@epfl.ch

- **Aleksandr Dremov**
  - SCIPER: 387716
  - Email: aleksandr.dremov@epfl.ch

## Abstract

Unlike SGD, methods with adaptive step sizes — such as Adam — are essential for training modern deep learning models, especially large language models.
Typically, the noise in the stochastic gradients is heavy-tailed for the latter.
Gradient clipping helps achieve good high-probability convergence for such noises.
Moreover, clipping fixes the provably poor high-probability convergence of Adam and AdaGrad.
In this project, we show that the distribution of gradient norms for NLP problems is significantly heavy-tailed, whereas this is not the case for computer vision settings.
We then investigate how different types of clipping affect model convergence in these two settings.
Our results demonstrate that clipping largely improves training under heavy-tailed noise scenarios, while it is not critical when the noise is sub-Gaussian.

## Code

Code is split into two parts: llm-based code and resnet-based code. They refer to the same repository, but different branches. We made such choice for reproducibility reasons as some changes of resnet experiments may be breaking.

We provide bash file `run.sh` that should reproduce the reported results.

To fetch git submodules code, use

```
git submodule update --init
```