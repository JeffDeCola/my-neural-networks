# THE MATH BEHIND TRAINING MLP NEURAL NETWORKS

_The example I used for my cheat sheet
[the-math-behind-training-mlp-neural-networks](https://github.com/JeffDeCola/my-cheat-sheets/blob/master/software/development/software-architectures/artificial-intelligence/artificial-intelligence-cheat-sheet/the-math-behind-training-mlp-neural-networks.md)
using my multi-layer perceptron
([mlp](https://github.com/JeffDeCola/my-go-packages/tree/master/neural-networks/mlp))
neural network go package._

Table of Contents

* [OVERVIEW](https://github.com/JeffDeCola/my-neural-networks/tree/main/perceptron-simple-example#overview)

Documentation and Reference

* [artificial intelligence](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/artificial-intelligence-cheat-sheet)
* [neural networks](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/artificial-intelligence-cheat-sheet/neural-networks-cheat-sheet.md)

## OVERVIEW

This is the example I used in my cheat sheet as a way to verify
the math.

```go
// Neural Network Parameters
nnp := mlp.NeuralNetworkParameters{
  InputNodes:              2,
  InputNodeLabels:         []string{"x[0]", "x[1]"},
  HiddenLayers:            1,
  HiddenNodesPerLayer:     []int{3},
  OutputNodes:             1,
  OutputNodeLabels:        []string{"y[0]"},
  LearningRate:            0.1,
  Epochs:                  10,
  DatasetCSVFile:          "dataset.csv",
  Initialization:          "file", // "random" or "file"
  WeightsAndBiasesCSVFile: "weights-and-biases.csv",
  MinMaxInput:             []float64{0.0, 100.0, 0.0, 100.0, 0.0, 100.0},
  MinMaxOutput:            []float64{0.0, 100.0},
  UseMinMaxInput:          false,
  UseMinMaxOutput:         true,
  NormalizeInputData:      true,
  NormalizeOutputData:     true,
  NormalizeMethod:         "zero-to-one",        // "zero-to-one" or "minus-one-to-one"
  ActivationFunction:      "sigmoid",            // "sigmoid" or "tanh"
  LossFunction:            "mean-squared-error", // "mean-squared-error" or "cross-entropy"
}
```