package main

import (
	"fmt"

	mlp "my-go-packages/mlp"
)

func main() {

	// Neural Network Parameters
	nnp := mlp.NeuralNetworkParameters{
		Mode:                         "training", // "training", "testing" or "predicting"
		InputNodes:                   2,
		InputNodeLabels:              []string{"x[0]", "x[1]"},
		HiddenLayers:                 2,             // Also update HiddenNodesPerLayer
		HiddenNodesPerLayer:          []int{15, 15}, // If using a file, must match
		OutputNodes:                  1,
		OutputNodeLabels:             []string{"y[0]"},
		Epochs:                       5000,
		LearningRate:                 0.1,
		ActivationFunction:           "sigmoid",                     // "sigmoid" or "tanh"
		LossFunction:                 "mean-squared-error",          // "mean-squared-error"
		InitWeightsBiasesMethod:      "file",                        // "file" or "random"
		InitWeightsBiasesJSONFile:    "trained-weights-biases.json", // ???????????????????????????????????
		TrainingDatasetCSVFile:       "training-dataset.csv",
		MinMaxInputMethod:            "calculate",   // "file" or "calculate" - Calculate from TrainingDatasetCSVFile
		MinMaxOutputMethod:           "calculate",   // "file" or "calculate" - Calculate from TrainingDatasetCSVFile
		MinMaxJSONFile:               "minmax.json", // Can be used for both training and testing
		NormalizeInputData:           true,
		NormalizeOutputData:          true,
		NormalizeMethod:              "zero-to-one", // "zero-to-one" or "minus-one-to-one"
		TrainedWeightsBiasesJSONFile: "trained-weights-biases.json",
		TestingDatasetCSVFile:        "testing-dataset.csv",
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
	nn.PrintDatasetMinMax()

	// Print the neural network
	fmt.Println("\nPrint the neural network")
	nn.PrintNeuralNetwork()

	// Train the neural network with dataset from a CSV file
	/*fmt.Println("\nTrain the neural network with dataset from a CSV file")
	err = nn.TrainNeuralNetwork()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}*/

	// Save the weights and biases to a json file
	// It will overwrite the file if it already exists
	/*fmt.Println("\nSave the weights and biases to a json file")
	err = nn.SaveWeightsBiasesToJSON()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}*/

	// Save the min and max values to a json file
	// It will overwrite the file if it already exists
	/*fmt.Println("\nSave the min and max values to a json file")
	err = nn.SaveMinMaxToJSON()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}*/

	// Test the neural network with the trained weights and biases
	// fmt.Println("\nRun the neural network with the trained weights and biases")
	// Read in the weights and biases from a json file
	data := []float64{80, 70, 73}
	err = nn.TestingNeuralNetwork(data)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

}
