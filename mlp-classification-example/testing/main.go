package main

import (
	"fmt"

	mlp "my-go-packages/mlp"
)

func main() {

	// Neural Network Parameters
	nnp := mlp.NeuralNetworkConfiguration{
		Mode:                         "testing", // "training", "testing" or "predicting"
		InputNodes:                   2,
		InputNodeLabels:              []string{"x[0]", "x[1]"},
		HiddenLayers:                 2,
		HiddenNodesPerLayer:          []int{15, 15},
		OutputNodes:                  1,
		OutputNodeLabels:             []string{"y[0]"},
		Epochs:                       1,
		LearningRate:                 0.1,
		ActivationFunction:           "sigmoid",            // "sigmoid" or "tanh"
		LossFunction:                 "mean-squared-error", // "mean-squared-error"
		InitWeightsBiasesMethod:      "file",               // "file" or "random"
		InitWeightsBiasesJSONFile:    "../training/trained-weights-biases.json",
		TrainingDatasetCSVFile:       "N/A",
		MinMaxInputMethod:            "file", // "file" or "calculate" from dataset File
		MinMaxOutputMethod:           "file", // "file" or "calculate" from dataset File
		MinMaxJSONFile:               "../training/minmax.json",
		NormalizeInputData:           true,
		NormalizeOutputData:          true,
		NormalizeMethod:              "zero-to-one", // "zero-to-one" or "minus-one-to-one"
		TrainedWeightsBiasesJSONFile: "N/A",
		TestingDatasetCSVFile:        "testing-dataset.csv",
		PredictingDatasetCSVFile:     "N/A",
	}

	// Create a new neural network
	fmt.Println("\nCreate a new neural network")
	nn := nnp.CreateNeuralNetwork()

	// Initialize the neural network (weights and bios with numbers from -1 to 1)
	// Will chose between random or file initialization
	fmt.Println("\nInitialize the neural network")
	err := nn.InitializeNeuralNetwork()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Get the input min and max values from the CSV file
	fmt.Println("\nGet the input min and max values from the CSV file")
	err = nn.SetMinMaxValues()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Print the input min and max values
	fmt.Println("\nPrint the input min and max values")
	nn.PrintMinMaxValues()

	// Print the neural network
	fmt.Println("\nPrint the neural network")
	nn.PrintNeuralNetwork()

	// Test the neural network with the trained weights and biases
	// fmt.Println("\nRun the neural network with the trained weights and biases")
	// Read in the weights and biases from a json file
	err = nn.TestNeuralNetwork()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

}
