
---

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/

we also give an example for distribute training on cifar10 dataset

执行命令示例
ps 节点执行： 

CUDA_VISIBLE_DEVICES='' python cifar10_distribute.py  --job_name=ps --task_index=0


worker 节点执行:

CUDA_VISIBLE_DEVICES=0 python cifar10_distribute.py  --job_name=worker --task_index=0

CUDA_VISIBLE_DEVICES=1 python cifar10_distribute.py  --job_name=worker --task_index=1

CUDA_VISIBLE_DEVICES=2 python cifar10_distribute.py  --job_name=worker --task_index=2

