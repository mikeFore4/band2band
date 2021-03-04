# Adaptive AutoEncoders for Class Conditional Image to Image Translation

## Overview
This repo contains a novel approach to image to image translation, specifically to be applied to the problem of translating images from one band of HLS data to another (for example, near infrared to the cirrus band). In this data, there are 14 distinct bands.

In scenarios where one would like to have an image to image translation model for translating images from multiple domains to multiple domains, the current best approach would require training a separate model for each pairwise combination of domains. As the number of domains expands, this approach becomes highly inefficient. Although class conditional generative models exist, to our knowledge, all class conditional models generate new images from scratch, rather than translating existing images across domains. Although real-time style transfer models also exist, these models only create plausible scenes in the other domain, while maintaining content from the source domain. These models do not offer the precision necessary fro the band to band translation problem as the level of detail necessary for accurate translation requires that the model learn information about each band during training rather than simply taking color/contrast information at inference time.

Our method is similar to the concept of [adaptive instance normalization](https://vision.cornell.edu/se3/wp-content/uploads/2017/08/adain.pdf). However, rather than learning a model to infer denormalization parameters from an input image at inference time, our method learns a separate embedding for each class during training. In this sense, the denormalization parameters are not fully adaptive, but rather conditional upon a fixed set of classes, or which any class can be selected at inference time.

## Training
Training uses torch's DistributedDataParallel module to enable training across multiple nodes and/or gpus. Training requires only a yaml config file and can be started via:
```
python -m torch.distributed.launch --nprocs_per_node=<num_gpus_per_node> train.py --config-file=<path/to/config.yaml>
```
An example config file is given in config.yaml
