# ML-term-proj

Machine learning term project adversial attack and decision boundary part.

## Introduction

Measuring the similarity between different initialized neural networks (with the same architecture) is essential to network interpretability. Here we consider neural networks as black boxes and investigate if networks' outputs are similar given the same inputs. Similar accuracy in only the test datapoints does not indicate that outputs would be similar in the **neighborhood space** of the datapoints. Therefore, we would like to investigate the similarity of the decision boundary/ decision "maps" in the neighborhood space between two different initialized neural networks. We first adopted adversial attack methods to measure **local** similarity. Then we did further experiments to measure the **global** similarity of decision boundary by introducing some metrics and by visualize the decision maps in some data samples.

## Requirements

pytorch==1.0.0

torchvision==0.2.1

cuda 8.0

## Usage

#### dataset

We use CIFAR 10 as our experiment dataset. By default, the CIFAR 10 dataset would be automatically downloaded into your folder `./data/`.

#### Train 5 different initialized ResNet 18

You can easily train different initialized ResNet 18 from scratch. Simply run

```
bash run.sh
```

The checkpoints will be stored in `./checkpoint/`. We also provide our pretrained model in `./checkpoint` folder in our repository (40MB each).

#### Reproduce our experimental results

All our experiments are in `experiment.ipynb`. This jupyter notebook contains detailed explanations for our experiments. You can reproduce our experiment simply by running each cell in order. You can also directly see our results in the outputs of each cell.

