package main

import (
    "fmt"
    "math/rand"
    "time"
)

// Perceptron struct
type Perceptron struct {
    weights []float64
    bias    float64
}

// NewPerceptron creates a new Perceptron with random weights and bias
func NewPerceptron(inputSize int) *Perceptron {
    rand.Seed(time.Now().UnixNano())
    weights := make([]float64, inputSize)
    for i := range weights {
        weights[i] = rand.Float64()*2 - 1 // Initialize weights between -1 and 1
    }
    bias := rand.Float64()*2 - 1 // Initialize bias between -1 and 1
    return &Perceptron{weights: weights, bias: bias}
}

// Activation function (step function)
func (p *Perceptron) Activation(sum float64) float64 {
    if sum >= 0 {
        return 1
    }
    return 0
}

// Predict makes a prediction based on input features
func (p *Perceptron) Predict(inputs []float64) float64 {
    sum := p.bias
    for i, input := range inputs {
        sum += input * p.weights[i]
    }
    return p.Activation(sum)
}

// Train trains the Perceptron using the given dataset
func (p *Perceptron) Train(dataset [][]float64, labels []float64, epochs int, learningRate float64) {
    for epoch := 0; epoch < epochs; epoch++ {
        for i, inputs := range dataset {
            prediction := p.Predict(inputs)
            error := labels[i] - prediction
            for j := range p.weights {
                p.weights[j] += learningRate * error * inputs[j]
            }
            p.bias += learningRate * error
        }
    }
}

func main() {

    // Example dataset (AND logic gate)
    dataset := [][]float64{
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1},
    }
    labels := []float64{0, 0, 0, 1}

    // Create a Perceptron
    perceptron := NewPerceptron(2)

    // Train the Perceptron
    epochs := 100
    learningRate := 0.1
    perceptron.Train(dataset, labels, epochs, learningRate)

    // Test the Perceptron
    for _, inputs := range dataset {
        prediction := perceptron.Predict(inputs)
        fmt.Printf("Inputs: %v, Prediction: %v\n", inputs, prediction)
    }
}
