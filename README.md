# No Wrong Turns: The Simple Geometry Of Neural Networks Optimization Paths 

The link of manuscript in OpenReview is [here](https://openreview.net/forum?id=zlW9RsqdLg)

## Abstract
Understanding the optimization dynamics of neural networks is necessary for closing the gap between theory and practice. 
Stochastic first-order optimization algorithms are known to efficiently locate favorable minima in deep neural networks. This efficiency, however, contrasts with the non-convex and seemingly complex structure of neural loss landscapes. In this study, we delve into the fundamental geometric properties of sampled gradients along optimization paths. We focus on two key quantities, which appear in the restricted secant inequality and error bound.
Both hold high significance for first-order optimization. Our analysis reveals that these quantities exhibit predictable, consistent behavior throughout training, despite the stochasticity induced by sampling minibatches.
Our findings suggest that not only do optimization trajectories never encounter significant obstacles, but they also maintain stable dynamics during the majority of training. These observed properties are sufficiently expressive to theoretically guarantee linear convergence and prescribe learning rate schedules mirroring empirical practices. We conduct our experiments on image classification, semantic segmentation and language modeling across different batch sizes, network architectures, datasets, optimizers, and initialization seeds. We discuss the impact of each factor.
Our work provides novel insights into the properties of neural network loss functions, and opens the door to theoretical frameworks more relevant to prevalent practice.


## Prerequisites
The codes in this repository is intended to work on clusters with Slurm's job scheduler.

#### Environment

```
Python: 3.10.2
CUDA: 11.4
CUDNN: 8200
```


#### Python Libraries

```
Package                      Version
---------------------------- -------------------
attrs                        22.2.0
einops                       0.6.0
huggingface-hub              0.11.1
numpy                        1.23.0
pandas                       1.5.0
pip                          21.3.1
timm                         0.6.12
torch                        1.13.1
torchtext                    0.14.1
torchvision                  0.14.1
wandb                        0.13.7
```

## Download
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet-1K](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) / signup is required
- [WikiText-2](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)
- [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)

## Quick Start
1. Change python path in `exp/env/cluster_env.sh` and module which you need to load
2. Change wandb entity to yours for each shell scripts such as `exp/image_classification/cifar10/batchsize-cifar.sh`
3. Change path to Imagenet in `exp/image_classification/imagenet1k/run_imagenet.sh` for launching imagenet task
4. Start following shells

## Reproduction Experiment

#### Image Classification Tasks

To run the ablation study of the optimizer in the image classification task of CIFAR-10, execute the following shell.

```
./exp/image_classification/cifar10/opt-cifar.sh
```

#### Word Language Modeling Tasks

To run the ablation study of the batchszie in the word language model task of WIkiText-2, execute the following shell.

```
./exp/word_language_model/wikitext2/batchsize-wikitext2.sh
```

#### Segmentation Tasks

To run the ablation study of the model architecture type in the segementation task of Vaihingen, execute the following shell.

```
./exp/segmentation/vaihingen/model-arch-vaihingen.sh
```

## License
All codes for experiments are modifications of the codes provided by [PyTorch's official implementation](https://github.com/pytorch/examples) for image classification and language modeling tasks and [code of Audebert et al.](https://github.com/nshaud/DeepNetsForEO) for segmentation task.
The license for the official Pytorch implementation is the BSD-3-Clause, and the license for the segmentation task implementation is GPLv3.


