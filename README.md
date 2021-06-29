## In-Time-Over-Parameterization Official Pytorch implementation

<img src="https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization/blob/main/ITOP.png" width="350" height="200">

**Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training**<br>
Shiwei Liu, Lu Yin, Decebal Constantin Mocanu, Mykola Pechenizkiy<br>
https://arxiv.org/abs/2102.02887<br>

This code base is created by Shiwei Liu during his Ph.D. at Eindhoven University of Technology. The implementation is heavily based on Tim Dettmers's implemenation for experiments on the sparse momentum.

## Requirements

The library requires Python 3.6.7, PyTorch v1.0.1, and CUDA v9.0
You can download it via anaconda or pip, see [PyTorch/get-started](https://pytorch.org/get-started/locally/) for further information. 

## Training 
### CIFAR10/100
We provide the training codes In-Time Over-Parameterization (ITOP). 

To train models with SET-ITOP with a typical training time, run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

```

To train models with SET-ITOP with an extended training time, change the value of --multiplier (e.g., 5 times) and run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 5 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

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
To train ResNet-50 on ImageNet with RigL-ITOP, run the following command:
```
cd ImageNet

CUDA_VISIBLE_DEVICES=0,1 python $1multiproc.py --nproc_per_node 2 $1main.py --multiplier 1 --growth gradient --master_port 4545 -j5 -p 500 --arch resnet50 -c fanin --update_frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --epochs 100 --density 0.2 $2 ../../../data/ --save save/ITOP/
```
change path of data ../../../data/ to the saved imagenet directory before running.

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


