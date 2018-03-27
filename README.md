# Cifar10 Challenge
*The code for my Cifar10 assignment in the WASP course.*

There are two types of models: a DenseNet model and a NASNet model.

The pretrained models can be found in the [models folder](models).

## Requirements
All the code is written for Python 3.x and to run it you also need several
Python packages preferably along with Nvidia CUDA to train the models on the GPU.

To install the Python dependencies use:
```bash
pip install -r requirements.txt
```

## Training

To train the respective model use:

```bash
# DenseNet
python train.m

# NASNet
python train_nasnet.py
```

## Testing

To test and evaluate the model use:

```bash
python test.m <model or directory of models>

# To test an ensemble of models that vote use
python test_ensemble.m <directory of models>
```

## References
- [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
- [DenseNet Author's Lua Implementation](https://github.com/liuzhuang13/DenseNet)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)
- [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)
- [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/pdf/1709.07417.pdf)
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/pdf/1608.03983.pdf)
- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012.pdf)
