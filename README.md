# Variational Neural Networks Pytorch

This repository contains a Pytorch implementation of Variational Neural Networks (VNNs) and image classification experiments for [Variational Neural Networks paper](https://arxiv.org/abs/2207.01524).

Bayesian Neural Networks (BNNs) provide a tool to estimate the uncertainty of a neural network by considering a distribution over weights and sampling different models for each input. In this paper, we propose a method for uncertainty estimation in neural networks called Variational Neural Network that, instead of considering a distribution over weights, generates parameters for the output distribution of a layer by transforming its inputs with learnable sub-layers. In uncertainty quality estimation experiments, we show that VNNs achieve better uncertainty quality than Monte Carlo Dropout or Bayes By Backpropagation methods.

## Run

Use `run_example.sh` to train and evaluate a single model on MNIST.
The corresponding reproducible capsule is available at [CodeOcean](https://codeocean.com/capsule/9585164/tree).
## Citation

If you use this work for your research, you can cite it as:
### Library:
```
@article{oleksiienko2022vnntorchjax,
    title = {Variational Neural Networks implementation in Pytorch and JAX},
    author = {Oleksiienko, Illia and Tran, Dat Thanh and Iosifidis, Alexandros},
    journal = {Software Impacts},
    volume = {14},
    pages = {100431},
    year = {2022},
}
```
### Paper:
```
@article{oleksiienko2023vnn,
  title={Variational Neural Networks}, 
  author = {Oleksiienko, Illia and Tran, Dat Thanh and Iosifidis, Alexandros},
  journal={arxiv:2207.01524}, 
  year={2023},
}
```
