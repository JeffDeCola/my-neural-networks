package main

import (
	"fmt"
	mlp "my-go-packages/neural-networks/mlp"

	logger "github.com/JeffDeCola/my-go-packages/golang/logger"
)

func main() {

	log := logger.CreateLogger(logger.LevelDebug, "jeff")

	// Neural Network Parameters
	nnp := mlp.NeuralNetworkConfiguration{
		Mode:                         "training", // "training", "testing" or "predicting"
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
		InitWeightsBiasesMethod:      "random",             // "file" or "random"
		InitWeightsBiasesJSONFile:    "N/A",
		TrainingDatasetCSVFile:       "training-dataset.csv",
		MinMaxInputMethod:            "calculate",   // "file" or "calculate" from dataset File
		MinMaxOutputMethod:           "calculate",   // "file" or "calculate" from dataset File
		MinMaxJSONFile:               "minmax.json", //from SaveMinMaxValuesToJSON()
		NormalizeInputData:           true,
		NormalizeOutputData:          true,
		NormalizeMethod:              "zero-to-one",                 // "zero-to-one" or "minus-one-to-one"
		TrainedWeightsBiasesJSONFile: "trained-weights-biases.json", // from SaveWeightsBiasesToJSON()
		TestingDatasetCSVFile:        "N/A",
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
		log.Error("Error: %s", err)
		return
	}

	// Get the input min and max values from the CSV file
	fmt.Println("\nGet the input min and max values from the CSV file")
	err = nn.SetMinMaxValues()
	if err != nil {
		log.Error("Error: %s", err)
		return
	}

	// Print the input min and max values
	fmt.Println("\nPrint the input min and max values")
	nn.PrintMinMaxValues()

	// Print the neural network
	fmt.Println("\nPrint the neural network")
	nn.PrintNeuralNetwork()

	// Train the neural network with dataset from a CSV file
	fmt.Println("\nTrain the neural network with dataset from a CSV file")
	err = nn.TrainNeuralNetwork()
	if err != nil {
		log.Error("Error: %s", err)
		return
	}

	// Save the min and max values to a json file
	// It will overwrite the file if it already exists
	fmt.Println("\nSave the min and max values to a json file")
	err = nn.SaveMinMaxValuesToJSON()
	if err != nil {
		log.Error("Error: %s", err)
		return
	}

	// Save the weights and biases to a json file
	// It will overwrite the file if it already exists
	fmt.Println("\nSave the weights and biases to a json file")
	err = nn.SaveWeightsBiasesToJSON()
	if err != nil {
		log.Error("Error: %s", err)
		return
	}

	log.Debug("This is a debug message") // Won't show (INFO level)
	log.Info("This is a Info Message", "env", "production", "user", "jeff")
	log.Warning("This is a Warning Message", 500)
	log.Error("This is an Error message")
	// log.Fatal("Done")

	// Dynamically change log level
	fmt.Printf("\nCHANGE LEVEL\n\n")
	log.ChangeLogLevel(logger.LevelDebug)

	log.Debug("This is a debug message") // Won't show (INFO level)
	log.Info("This is a Info Message", "env", "production", "user", "jeff")
	log.Warning("This is a Warning Message", 500)
	log.Error("This is an Error message")

}
