package main

import (
	"fmt"
	"os"

	logger "my-go-packages/golang/logger"
	mlp "my-go-packages/neural-networks/mlp"
)

func main() {

	log := logger.CreateLogger(logger.Debug, "jeffs_noTime", os.Stdout)

	// To gain insight on the inner working of the mlp
	// You may set logging level for mlp package
	mlp.ShowDebug()

	// Neural Network Parameters
	nnp := mlp.NeuralNetworkConfiguration{
		Mode:                         "testing", // "training", "testing" or "predicting"
		InputNodes:                   2,
		InputNodeLabels:              []string{"x[0]", "x[1]"},
		HiddenLayers:                 2,
		HiddenNodesPerLayer:          []int{15, 15},
		OutputNodes:                  1,
		OutputNodeLabels:             []string{"y[0]"},
		Epochs:                       0,
		LearningRate:                 0,
		ActivationFunction:           "sigmoid", // "sigmoid" or "tanh"
		LossFunction:                 "N/A",     // "mean-squared-error"
		InitWeightsBiasesMethod:      "file",    // "file" or "random"
		InitWeightsBiasesJSONFile:    "../mlp-class-training/trained-weights-biases.json",
		MinMaxInputMethod:            "file", // "file" or "calculate"
		MinMaxOutputMethod:           "file", // "file" or "calculate"
		MinMaxJSONFile:               "../mlp-class-training/minmax.json",
		NormalizeInputData:           true,
		NormalizeOutputData:          true,
		NormalizeMethod:              "zero-to-one", // "zero-to-one" or "minus-one-to-one"
		TrainingDatasetCSVFile:       "N/A",
		TestingDatasetCSVFile:        "testing-dataset.csv",
		PredictingDatasetCSVFile:     "N/A",
		TrainedWeightsBiasesJSONFile: "N/A",
	}

	// Create a new neural network
	fmt.Println("\nCreate a new neural network")
	nn := nnp.CreateNeuralNetwork()

	// Initialize the neural Network from the trained Weights Biases JSON File
	fmt.Println("\nInitialize the neural network")
	err := nn.InitializeNeuralNetwork()
	if err != nil {
		log.Error("Initialization failed", "error", err)
		return
	}

	// Get the input min and max values from the training CSV file
	fmt.Println("\nGet the input min and max values from the CSV file")
	err = nn.SetMinMaxValues()
	if err != nil {
		log.Error("Set min max values failed", "error", err)
		return
	}

	// Test the neural network with the trained weights and biases
	fmt.Println("\nRun the neural network with the trained weights and biases")
	err = nn.TestNeuralNetwork([]float64{45.00, 55.00, 52.00})
	if err != nil {
		log.Error("Test Neural NetWork Failed", "error", err)
		return
	}

	// Print the neural network
	fmt.Println("\nPrint the neural network")
	nn.PrintNeuralNetwork()

	// Print the input min and max values
	fmt.Println("\nPrint the input min and max values")
	nn.PrintMinMaxValues()

}
