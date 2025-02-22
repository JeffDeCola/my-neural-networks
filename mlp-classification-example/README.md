# MLP CLASSIFICATION EXAMPLE

_Using my multi-layer perceptron (MLP) neural network go package for
data classification._

Table of Contents

* [PERCEPTRON](https://github.com/JeffDeCola/my-neural-networks#perceptron)
* [MULTI-LAYER PERCEPTRON (MLP)](https://github.com/JeffDeCola/my-neural-networks#multi-layer-perceptron-mlp)

Documentation and Reference

* [Artificial intelligence cheat sheet](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/artificial-intelligence-cheat-sheet)
* [Neural networks cheat sheet](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/artificial-intelligence-cheat-sheet/neural-networks.md)

## OVERVIEW

This project implements a multi-layer perceptron (MLP) neural network designed
to predict student performance based on three input features:

* midterm grade
* hours studied
* last test grade

The network consists of three hidden layers, each containing four nodes,
and produces two outputs:

* the percentage likelihood of passing the final test and
* the predicted final grade.

It will have the following structure,

* Input Data:  $i_{[1]}$, $i_{[2]}$, $i_{[3]}$
* Input Nodes: 3
* Hidden Nodes: 4
* Output Nodes: 2
* Target Data: $z_{[0]}$, $z_{[1]}$
* Activation Function: Sigmoid $f(s)$



## TRAINING DATA

We will be using the following training data to train our neural network,

$$
\begin{bmatrix}
i_{[0]} & i_{[1]} & i_{[2]} & z_{[0]} & z_{[1]} \\
4 & 4 & 3 & 4 & 4 \\
3 & 48 & 63 & 66 & 34 \\
42 & 22 & 32 & 42 & 41 \\
. & . & . & . & . \\
etc. & etc. & etc. & etc. & etc.
\end{bmatrix}
$$