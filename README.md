# VGG-Net
 VGG family of Convolutional Neural Networks. A CNN can be considered VGG-net like if:
1. It makes use of only 3 × 3 filters, regardless of network depth.
2. There are multiple CONV => RELU layers applied before a single POOL operation, some-
times with more CONV => RELU layers stacked on top of each other as the network increases
in depth.
We then implemented a VGG inspired network, suitably named MiniVGGNet. This network
architecture consisted of two sets of (CONV => RELU) * 2) => POOL layers followed by an FC => RELU => FC => SOFTMAXlayerset.Wealsoappliedbatchnormalizationaftereveryactivation as well as dropout after every pool and fully-connected layer. To evaluate MiniVGGNet, we used the CIFAR-10 dataset.

Fire up your terminal shell - "python minivggnet_cifar10.py \
--output output/cifar10_minivggnet_without_bn.png" in the desired folder
