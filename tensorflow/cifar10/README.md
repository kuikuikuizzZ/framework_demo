
---

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/

we also give an example for distribute training on cifar10 dataset

### step 1:
we need to modify our **ps_host** and **worker_host** in cifar10_distributed.py  for your cluster:
```pyhon 
flags.DEFINE_string('ps_hosts', '172.17.0.9:22221', 'Comma-separated list of hostname:port pairs')
# 三个worker节点
flags.DEFINE_string('worker_hosts', '172.17.0.12:22221,172.17.0.13:22221,172.17.0.14:22221',
                    'Comma-separated list of hostname:port pairs')
```
or add arguments **--ps_hosts=0.0.0.0:10000** **--worker_hosts=0.0.0.0:10000** in your setup command

### step2:
run the python script on each note, make sure the data_dir is correct
#### example:

ps note command： 

CUDA_VISIBLE_DEVICES='' python cifar10_distributed.py  --job_name=ps --task_index=0

worker command: 

CUDA_VISIBLE_DEVICES=0 python cifar10_distributed.py  --job_name=worker --task_index=0

CUDA_VISIBLE_DEVICES=1 python cifar10_distributed.py  --job_name=worker --task_index=1

CUDA_VISIBLE_DEVICES=2 python cifar10_distributed.py  --job_name=worker --task_index=2


