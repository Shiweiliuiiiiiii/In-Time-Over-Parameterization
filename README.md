# In-Time-Over-Parameterization Official Pytorch implementation

<img src="https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization/blob/main/ITOP.png" width="700" height="400">

**Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training**<br>
Shiwei Liu, Lu Yin, Decebal Constantin Mocanu, Mykola Pechenizkiy<br>
https://arxiv.org/abs/2102.02887<br>

Abstract: *In this paper, we introduce a new perspective on training deep neural networks capable of state-of-the-art performance without the need for the expensive over-parameterization by proposing the concept of In-Time Over-Parameterization (ITOP) in sparse training. By starting from a random sparse network and continuously exploring sparse connectivities during training, we can perform an Over-Parameterization in the space-time manifold, closing the gap in the expressibility between sparse training and dense training. We further use ITOP to understand the underlying mechanism of Dynamic Sparse Training (DST) and indicate that the benefits of DST come from its ability to consider across time all possible parameters when searching for the optimal sparse connectivity. As long as there are sufficient parameters that have been reliably explored during training, DST can outperform the dense neural network by a large margin. We present a series of experiments to support our conjecture and achieve the state-of-the-art sparse training performance with ResNet-50 on ImageNet. More impressively, our method achieves dominant performance over the overparameterization-based sparse methods at extreme sparsity levels. When trained on CIFAR-100, our method can match the performance of the dense model even at an extreme sparsity (98%).*

This code base is created by Shiwei Liu [s.liu3@tue.nl](mailto:s.liu3@tue.nl) during his Ph.D. at Eindhoven University of Technology.<br>

The implementation is heavily based on Tim Dettmers' implemenation for experiments on the sparse momentum.

## Requirements

The library requires Python 3.6.7, PyTorch v1.0.1, and CUDA v9.0
You can download it via anaconda or pip, see [PyTorch/get-started](https://pytorch.org/get-started/locally/) for further information. 


## Training 
Our implementation includes the code for two dynamic sparse training methods SET (https://www.nature.com/articles/s41467-018-04316-3) and RigL (https://arxiv.org/abs/1911.11134). The main difference is the weight regorwing method: using --growth random for SET; using --growth gradient for RigL.


### CIFAR10/100
We provide the training codes for In-Time Over-Parameterization (ITOP). 

To train a **dense model**, we just need to remove the --sparse argument.

```
python main.py --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

```
To train models with **SET-ITOP** with a **typical** training time, run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

```
To train models with **RigL-ITOP** with a **typical** training time, run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 4000 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth gradient --death magnitude --redistribution none

```

To train models with **SET-ITOP** with an **extended** training time, change the value of --multiplier (e.g., 5 times) and run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 5 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

```
To train models with **RigL-ITOP** with an **extended** training time, change the value of --multiplier (e.g., 5 times) and run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 5 --lr 0.1 --density 0.05 --update_frequency 4000 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth gradient --death magnitude --redistribution none

```

Options:
* --sparse - Enable sparse mode (remove this if want to train dense model)
* --sparse_init - type of sparse initialization. Choose from: uniform, ERK
* --model (str) - type of networks
```
  MNIST:
	lenet5
	lenet300-100

 CIFAR-10/100：
	alexnet-s
	alexnet-b
	vgg-c
	vgg-d
	vgg-like
	wrn-28-2
	wrn-22-8
	wrn-16-8
	wrn-16-10
  ResNet-34
```
* --growth (str) - growth mode. Choose from: random, gradient, momentum
* --death (str) - removing mode. Choose from: magnitude, SET, threshold
* --redistribution (str) - redistribution mode. Choose from: magnitude, nonzeros, or none. (default none)
* --density (float) - density level (default 0.05)
* --death-rate (float) - initial pruning rate (default 0.5)

The sparse operatin is in the sparsetraining/core.py file. 

For better sparse training performance, it is suggested to decay the learning rate at the 1/2 and 3/4 training time instead of using the default learning rate schedule in main.py. 

### ImageNet with ResNet-50
To train ResNet-50 on ImageNet with **RigL-ITOP**, run the following command:
```
cd ImageNet

CUDA_VISIBLE_DEVICES=0,1 python $1multiproc.py --nproc_per_node 2 $1main.py --multiplier 1 --growth gradient --master_port 4545 -j5 -p 500 --arch resnet50 -c fanin --update_frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --epochs 100 --density 0.2 $2 ../../../data/ --save save/ITOP/
```
change path of data ../../../data/ to the saved imagenet directory before running.

Results on ImageNet

### 1x training run

| Methods            | Sparsity |   Top-1 Acc   | Rs  | Training FLOPs | Test FLOPs |
| -------------------|----------|---------------|-----|----------------| ---------- |
| Dense | 0.0 |  76.8      |        1.0         |      1x(3.2e18)      | 1x(8.2e9)  |
| RigL  | 0.8 |  75.1      |         -          |      0.42x           | 0.42x      | 
| RigL-ITOP  | 0.8 |  75.8      |         0.93          |      0.42x           | 0.42x      | 

| Methods            | Sparsity |   Top-1 Acc   | Rs  | Training FLOPs | Test FLOPs |
| -------------------|----------|---------------|-----|----------------| ---------- |
| Dense | 0.0 |  76.8      |        1.0         |      1x(3.2e18)      | 1x(8.2e9)  |
| RigL  | 0.9 |  73.0      |         -          |      0.25x           | 0.24x      | 
| RigL-ITOP  | 0.9 |  73.8      |         0.83          |      0.25x           | 0.24x      | 

### extended training run

| Methods            | Sparsity |   Top-1 Acc   | Rs  | Training FLOPs | Test FLOPs |
| -------------------|----------|---------------|-----|----------------| ---------- |
| RigL (5 times)  | 0.8 |  77.1    |         -          |     2.09x           | 0.42x      | 
| RigL-ITOP (2 times)  | 0.8 |  76.9      |         0.97          |      0.84x           | 0.42x      | 

| Methods            | Sparsity |   Top-1 Acc   | Rs  | Training FLOPs | Test FLOPs |
| -------------------|----------|---------------|-----|----------------| ---------- |
| RigL (5 times)  | 0.9 |  76.4      |         -          |      0.25x           | 0.24x      | 
| RigL-ITOP (2 times) | 0.9 |    75.5    |         0.89          |      0.50x           | 0.24x      | 

## Other Implementations
[One million neurons](https://github.com/Shiweiliuiiiiiii/SET-MLP-ONE-MILLION-NEURONS): truly sparse SET implementation with cpu!

[Sparse Evolutionary Training](https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks): official SET implementation with Keras.

## Citation
If you use this library in a research paper, please cite this repository.
```
@article{liu2021we,
  title={Do we actually need dense over-parameterization? in-time over-parameterization in sparse training},
  author={Liu, Shiwei and Yin, Lu and Mocanu, Decebal Constantin and Pechenizkiy, Mykola},
  journal={arXiv preprint arXiv:2102.02887},
  year={2021}
}
```


## Table of Contents
More information is coming soon.


