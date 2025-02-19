# MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORK

_This is an implementation of a user scalable multi-layer
perceptron (MLP) / feed-forward (FF) neural network using a
sigmoid non-linear function written in go._

Table of Contents

* [OVERVIEW](#OVERVIEW)
* [TRAINING DATA](#TRAINING-DATA)
* [TRAINING (BACKPROPAGATION)](#TRAINING-BACKPROPAGATION)
  * [STEP 1 - FORWARD PASS](#STEP-1---FORWARD-PASS)
  * [STEP 2 - BACKWARD PASS](#STEP-2---BACKWARD-PASS)
  * [STEP 3 - UPDATE WEIGHTS](#STEP-3---UPDATE-WEIGHTS)

Documentation and Reference

* [artificial intelligence cheat sheet](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/artificial-intelligence-cheat-sheet)
* [neural networks cheat sheet](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/artificial-intelligence-cheat-sheet/neural-networks.md)

## OVERVIEW

This multi-layer perceptron (MLP) / feed-forward (FF)
neural network will be used to classify data into two categories.

This project implements a multi-layer perceptron (MLP) neural network designed
to predict student performance based on three input features:
* midterm grade, 
* hours studied, and 
* last test grade.

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

![IMAGE mathematical model of a neuron - IMAGE](./feed-forward-multi-layer-perceptron-neural-network.svg)

## TRAINING DATA

We will be using the following [training data]() to train our neural network,

$$
\begin{bmatrix}
\mathbf{i_{[0]}} & \mathbf{i_{[1]}} & \mathbf{i_{[2]}} & \mathbf{z_{[0]}} & \mathbf{z_{[1]}} \\
4 & 4 & 3 & 4 & 4 \\
3 & 48 & 63 & 66 & 34 \\
42 & 22 & 32 & 42 & 41 \\
. & . & . & . & . \\
etc... & etc... & etc... & etc... & etc... \\
\end{bmatrix}
$$

## TRAINING (BACKPROPAGATION)

A training data set for the neural
network on a set of training data.
This is were the neural network can _learn_
(adjusting the weights) on training data
in order to map the inputs to outputs.
The training process is called backpropagation which is
a supervised learning algorithm. This means,
for some given inputs, we know the desired/expected target output.

### STEP 1 - FORWARD PASS

Giving our $x_{[0]}$, $x_{[1]}$ and $x_{[2]}$ input training data,
**compute the output for each layer and
propagate through layers to obtain the outputs
$y_{[0]}$ and $y_{[1]}$**.

![IMAGE feed-forward-multi-layer-perceptron-neural-network-training-step-1 - IMAGE](feed-forward-multi-layer-perceptron-neural-network-training-step-1.svg)

Normalize the input data between -1 and 1,

$$
\mathbf{x_{[0]}} = normalize(i_{[0]}) \\
\mathbf{x_{[1]}} = normalize(i_{[1]}) \\
\mathbf{x_{[2]}} = normalize(i_{[2]})
$$

Calculate the hidden layer outputs,

$$
\begin{aligned}
\mathbf{y_{h1[0]}} &= f_{h1[0]}(s) = f_{h1[0]}\left(x_{[0]} w_{h1[0]X[0]} + x_{[1]} w_{h1[0]X[1]} + x_{[2]} w_{h1[0]X[2]} + b_{h1[0]}\right) \\
\mathbf{y_{h1[1]}} &= f_{h1[1]}(s) = f_{h1[1]}\left(x_{[0]} w_{h1[1]X[0]} + x_{[1]} w_{h1[1]X[1]} + x_{[2]} w_{h1[1]X[2]} + b_{h1[1]}\right) \\
\mathbf{y_{h1[2]}} &= f_{h1[2]}(s) = f_{h1[2]}\left(x_{[0]} w_{h1[2]X[0]} + x_{[1]} w_{h1[2]X[1]} + x_{[2]} w_{h1[2]X[2]} + b_{h1[2]}\right) \\
\mathbf{y_{h1[3]}} &= f_{h1[3]}(s) = f_{h1[3]}\left(x_{[0]} w_{h1[3]X[0]} + x_{[1]} w_{h1[3]X[1]} + x_{[2]} w_{h1[3]X[2]} + b_{h1[3]}\right) \\

\end{aligned}
$$

Finally, calculate the outputs,

$$
\begin{aligned}
\mathbf{y_{[0]}} = y_{o[0]} &= f_{o[0]}(s) = f_{o[0]}\left(y_{h1[0]} w_{o[0]h1[0]} + y_{h1[1]} w_{o[0]h1[1]} + y_{h1[2]} w_{o[0]h1[2]} + y_{h1[3]} w_{o[0]h1[3]} + b_{o[0]}\right) \\
\mathbf{y_{[1]}} = y_{o[1]} &= f_{o[1]}(s) = f_{o[1]}\left(y_{h1[0]} w_{o[1]h1[0]} + y_{h1[1]} w_{o[1]h1[1]} + y_{h1[2]} w_{o[1]h1[2]} + y_{h1[3]} w_{o[1]h1[3]} + b_{o[1]}\right) \\
\end{aligned}
$$

I'm actually trying to simplify the above equations.
It makes more sense to code it using slices.

### STEP 2 - BACKWARD PASS

Now  that we have the outputs $y$, calculate the error (delta **$\delta$**)
between target data ($z$) and actual output ($y$)
and propagate backwards.

![IMAGE feed-forward-multi-layer-perceptron-neural-network-training-step-2 - IMAGE](feed-forward-multi-layer-perceptron-neural-network-training-step-2.svg)

The output error,

$$
\begin{aligned}
\mathbf{\delta_{o[0]}} &= \delta_{[0]} = z_{[0]} - y_{[0]} \\
\mathbf{\delta_{o[1]}} &= \delta_{[1]} = z_{[1]} - y_{[1]} \\
\end{aligned}
$$

The hidden layer error,

$$
\begin{aligned}
\delta_{h1[0]} &= \delta_{h1[0]} w_{h1[0]} + \delta_{} w_{(21)32} \\
\delta_{h1[1]} &= \delta_{31} w_{(22)31} + \delta_{32} w_{(22)32} \\
\delta_{h1[2]} &= \delta_{31} w_{(23)31} + \delta_{32} w_{(23)32} \\
\delta_{h1[3]} &= \delta_{31} w_{(24)31} + \delta_{32} w_{(24)32} \\
\end{aligned}
$$

## STEP 3 - UPDATE WEIGHTS

Update the weights using the error (delta $\delta$) and the learning rate $\alpha$.

![IMAGE feed-forward-multi-layer-perceptron-neural-network-training-step-3 - IMAGE](feed-forward-multi-layer-perceptron-neural-network-training-step-3.svg)

The new weights,

$$
\begin{aligned}
w_{(21)31} &= w_{(21)31} + \alpha \delta_{1} y_{21} \\
w_{(22)31} &= w_{(22)31} + \alpha \delta_{1} y_{22} \\
w_{(23)31} &= w_{(23)31} + \alpha \delta_{1} y_{23} \\
w_{(24)31} &= w_{(24)31} + \alpha \delta_{1} y_{24} \\
w_{(21)32} &= w_{(21)32} + \alpha \delta_{2} y_{21} \\
w_{(22)32} &= w_{(22)32} + \alpha \delta_{2} y_{22} \\
w_{(23)32} &= w_{(23)32} + \alpha \delta_{2} y_{23} \\
w_{(24)32} &= w_{(24)32} + \alpha \delta_{2} y_{24} \\
\end{aligned}
$$
