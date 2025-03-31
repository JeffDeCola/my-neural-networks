package main

import (
	"fmt"

	logger "my-go-packages/golang/logger"
	mlp "my-go-packages/neural-networks/mlp"
)

func main() {

	log := logger.CreateLogger(logger.Debug, "jeffs_noTime")

	// To gain insight on the inner working of the mlp
	// You may set logging level for mlp package
	// mlp.ShowDebug()

	// Neural Network Parameters
	nnp := mlp.NeuralNetworkConfiguration{
		Mode:                         "training", // "training", "testing" or "predicting"
		InputNodes:                   2,
		InputNodeLabels:              []string{"x[0]", "x[1]"},
		HiddenLayers:                 2,
		HiddenNodesPerLayer:          []int{15, 15},
		OutputNodes:                  1,
		OutputNodeLabels:             []string{"y[0]"},
		Epochs:                       1000,
		LearningRate:                 0.1,
		ActivationFunction:           "sigmoid",            // "sigmoid" or "tanh"
		LossFunction:                 "mean-squared-error", // "mean-squared-error"
		InitWeightsBiasesMethod:      "random",             // "file" or "random"
		InitWeightsBiasesJSONFile:    "N/A",
		MinMaxInputMethod:            "calculate", // "file" or "calculate"
		MinMaxOutputMethod:           "calculate", // "file" or "calculate"
		MinMaxJSONFile:               "minmax.json",
		NormalizeInputData:           true,
		NormalizeOutputData:          true,
		NormalizeMethod:              "zero-to-one", // "zero-to-one" or "minus-one-to-one"
		TrainingDatasetCSVFile:       "training-dataset.csv",
		TestingDatasetCSVFile:        "N/A",
		PredictingDatasetCSVFile:     "N/A",
		TrainedWeightsBiasesJSONFile: "trained-weights-biases.json",
	}

	// Create a new neural network
	fmt.Println("\nCreate a neural network")
	nn := nnp.CreateNeuralNetwork()

	// Initialize the neural network (weights and bios with numbers from -1 to 1)
	// Will chose between random or file initialization
	fmt.Println("\nInitialize the neural network")
	err := nn.InitializeNeuralNetwork()
	if err != nil {
		log.Error("Initialization failed", "error", err)
		return
	}

	// Set the input min and max values
	fmt.Println("\nSet the input min and max values")
	err = nn.SetMinMaxValues()
	if err != nil {
		log.Error("Set min max values failed", "error", err)
		return
	}

	// Train the neural network with dataset from a CSV file
	fmt.Println("\nTrain the neural network with dataset from a CSV file")
	err = nn.TrainNeuralNetwork()
	if err != nil {
		log.Error("Train Neural NetWork Failed", "error", err)
		return
	}

	// Save the min and max values to a json file
	// It will overwrite the file if it already exists
	fmt.Println("\nSave the min and max values to a json file")
	err = nn.SaveMinMaxValuesToJSON()
	if err != nil {
		log.Error("Save min max values to json failed", "error", err)
		return
	}

	// Save the weights and biases to a json file
	// It will overwrite the file if it already exists
	fmt.Println("\nSave the weights and biases to a json file")
	err = nn.SaveWeightsBiasesToJSON()
	if err != nil {
		log.Error("Save the weights and biases failed", "error", err)
		return
	}

	// Print the neural network
	fmt.Println("\nPrint the neural network")
	nn.PrintNeuralNetwork()

	// Print the input min and max values
	fmt.Println("\nPrint the input min and max values")
	nn.PrintMinMaxValues()

	// Print Summary of Training
	fmt.Println("\nPrint the training summary")
	nn.PrintTrainingSummary()

}
