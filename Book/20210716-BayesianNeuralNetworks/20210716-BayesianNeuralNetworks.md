# Bayesian Neural Networks

## Tips

### What is the difference between BNNs and NNs?

Bayesian Neural Networks (BNNs) have two desirable advantages over ordinary neural networks that are point estimations trained with stochastic gradient descent.
First, an ordinary neural network cannot estimate the uncertainty of its output. As a result, silent failure can lead to dramatic outcomes.
In contrast, A BNN is capable of measuring such uncertainty.
The uncertainty can benefit downstream tasks and can also be used for out-of-distribution detection.
Second, A BNN can explicitly incorporate a prior distribution over its parameters, which gives promising direction for incremental learning. The posterior distribution of parameters after seen old-task data can be used as a priori in the next incremental-learning step.

### Concepts

A Bayesian neural network (BNN) is a stochastic artificial neural network trained using Bayesian inference.
Stochasticity can be imposed in two ways:
(1) consider the parameters in BNN with parameterized distributions instead of deterministic values;
(2) consider the activations in BNN as stochastic variables.

### A promising direction for BNN application in large-scale data

The stochastic variational inference might be a promising direction for BNN application in large-scale data.

### The major problem of BNN application in large-scale data

The major problem of BNN application in large-scale data is the intractable computational burden in computing the integration. The non-linearity of deep neural network architectures makes it impossible to derive a closed-form solution for the integration.

## A road map to 3D object detection for TIL

- [X] linear regression with BNN
- [X] logistic regression with BNN
- [ ] MNIST with BNN
- [ ] MNIST with BNN (TIL)
- [ ] MNIST-Segmentation with BNN
- [ ] MNIST-Segmentation with BNN (TIL)
- [ ] 3D object detection with BNN
- [ ] 3D object detection with BNN (TIL)

## A road map to Bayesian Neural Network for 3D object detection and TIL

BNN basics:
- [X] Hands-on Bayesian Neural Networks - A tutorial for deep learning users
- [X] Stochastic Variational Inference (SVI)
- [X] Hamiltonian Monte Carlo Inference (HMC)

BNN in large-scale data
- [X] Publications of Prof. Gal Yarin
- [X] Bayes-by-backpop

BNN in task-incremental learning
- [ ] Continual Learning using Bayesian Neural Networks
- [ ] Uncertainty-guided continual learning with Bayesian Neural Networks
