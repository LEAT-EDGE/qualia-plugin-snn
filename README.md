# Qualia-Plugin-SNN
Copyright 2023 © Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, LEAT. All rights reserved.

Plugin for Spiking Neural Network support inside Qualia.

## Install

```
git clone https://naixtech.unice.fr/gitlab/qualia/qualia-plugin-snn
cd qualia-plugin-snn
pdm venv create
pdm use "$(pwd)/.venv/bin/python"
$(pdm venv activate in-project)
pdm install -G gsc -G codegen
```

## Run CIFAR-10 SVGG16 example

Download [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and extract it inside `data/`.

```
qualia conf/cifar10/vgg16_bn_ifsr_float32_train.toml preprocess_data
qualia conf/cifar10/vgg16_bn_ifsr_float32_train.toml train
qualia conf/cifar10/vgg16_bn_ifsr_float32_train.toml prepare_deploy
qualia conf/cifar10/vgg16_bn_ifsr_float32_train.toml deploy_and_evaluate
```

## Acknowledgment

* [SpikingJelly](https://github.com/fangwei123456/spikingjelly) [^1]

[^1]: Please note article V.1 "Disclosure of Commercial Use" of the *Open-Intelligence Open Source License V1.0*
