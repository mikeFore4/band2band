# Denormalization Embedded AutoEncoders for Class Conditional Image to Image Translation

## Overview
This repo contains a novel approach to image to image translation, specifically to be applied to the problem of translating images from one band of HLS data to another (for example, near infrared to the cirrus band). In this data, there are 14 distinct bands.

In scenarios where one would like to have an image to image translation model for translating images from multiple domains to multiple domains, the current best approach would require training a separate model for each pairwise combination of domains. As the number of domains expands, this approach becomes highly inefficient. Although class conditional generative models exist, to my knowledge, all class conditional models generate new images from scratch, rather than translating existing images across domains. Although real-time style transfer models also exist, these models only create plausible scenes in the other domain, while maintaining content from the source domain. These models do not offer the precision necessary for the band to band translation problem as the level of detail necessary for accurate translation requires that the model learn information about each band during training rather than simply taking color/contrast information at inference time.

My method is similar to the concept of [adaptive instance normalization](https://vision.cornell.edu/se3/wp-content/uploads/2017/08/adain.pdf). However, rather than learning a model to infer denormalization parameters from an input image at inference time, my method learns a separate embedding for each class during training. In this sense, the denormalization parameters are not fully adaptive, but rather conditional upon a fixed set of classes, of which any class can be selected at inference time.

## Use
Training uses torch's DistributedDataParallel module to enable training across multiple nodes and/or gpus. Training requires only a yaml config file and can be started via:
```
python -m torch.distributed.launch --nprocs_per_node=<num_gpus_per_node> train_net.py --config-file=<path/to/config.yaml>
```
An example config file is given in config.yaml

Validation can be performed during training or separately. To perform a separate validation run, execute:
```
python -m torch.distributed.launch --nprocs_per_node=<num_gpus_per_node> test_net.py --config-file=<path/to/config.yaml>
```
For inference, you need to provide a folder containing all the input images which need to be of the same input class. Then perform the following:
```
python inference.py --config-file=<path/to/config.yaml> --input-dir=<path/to/input/images> --output-dir=<path/to/write/images> --input-class=<int> --output-class=<int>
```
This inference script can also be run in distributed mode with torch.distributed.launch. There is probably a better way to implement the inference script so it doesn't have to run separately for each class transition, but that would likely require an input dict from the user to map which images to translate to which classes. I will explore inputs like this at a later date.
