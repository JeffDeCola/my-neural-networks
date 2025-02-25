# MLP CLASSIFICATION EXAMPLE

_Using my multi-layer perceptron (MLP) neural network go package for
data classification._

Table of Contents

* []()
* []()

Documentation and Reference

* [Artificial intelligence cheat sheet](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/artificial-intelligence-cheat-sheet)
* [Neural networks cheat sheet](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/artificial-intelligence-cheat-sheet/neural-networks.md)
* My
 [mlp go package](https://github.com/JeffDeCola/my-go-packages/tree/master/mlp)
 I use for this example

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

Hence, I will start with the following parameters to build my neural network,

```go
nnp := mlp.NeuralNetworkParameters{
  InputNodes:          3,
  InputNodeLabels:     []string{"midterm-grade", "hours-studied", "last-test-grade"},
  HiddenLayers:        3,
  HiddenNodesPerLayer: []int{4, 4, 4},
  OutputNodes:         2,
  OutputNodeLabels:    []string{"pred-perc-passing-final", "pred-final-grade"},
  LearningRate:        0.1,
  Epochs:              4,
  DatasetCSVFile:      "dataset.csv",
}
```

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

## EXAMPLE

This is a main go code I used,

```go

```
