package main

import (
	"fmt"

	mlp "my-go-packages/mlp"
)

func main() {

	// Neural Network Parameters
	nnp := mlp.NeuralNetworkParameters{
		InputNodes:              2,
		InputNodeLabels:         []string{"midterm-grade", "hours-studied", "last-test-grade"},
		HiddenLayers:            1,
		HiddenNodesPerLayer:     []int{3},
		OutputNodes:             1,
		OutputNodeLabels:        []string{"predicted-percentage-passing-final", "predicted-final-grade"},
		LearningRate:            0.1,
		Epochs:                  1,
		DatasetCSVFile:          "dataset.csv",
		Initialization:          "file", // "random" or "file"
		WeightsAndBiasesCSVFile: "weights-and-biases.csv",
		NormalizeInputData:      true,
		NormalizeMethod:         "zero-to-one",        // "zero-to-one" or "minus-one-to-one"
		ActivationFunction:      "sigmoid",            // "sigmoid" or "tanh"
		LossFunction:            "mean-squared-error", // "mean-squared-error" or "cross-entropy"
	}

	// Create a new neural network
	fmt.Println("\nCREATE NEURAL NETWORK --------------------------------------")
	nn := nnp.CreateNeuralNetwork()

	// Get the input min and max values from the CSV file
	fmt.Println("\nGET INPUT MIN AND MAX VALUES FROM CSV FILE -----------------")
	err := nn.GetInputMinMaxFromCSV()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Print the input min and max values
	fmt.Println("\nPRINT INPUT MIN AND MAX VALUES -----------------------------")
	nn.PrintInputMinMax()

	// Initialize the neural network (weights and bios with numbers from -1 to 1)
	// Will chose between random or file initialization
	fmt.Println("\nINITIALIZE NEURAL NETWORK ----------------------------------")
	err = nn.InitializeNeuralNetwork()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Print the neural network
	fmt.Println("\nPRINT NEURAL NETWORK ---------------------------------------")
	nn.PrintNeuralNetwork()

	// Train the neural network with dataset from CSV file
	fmt.Println("\nTHE TRAINING LOOP ---------------------------------------")
	err = nn.TrainNeuralNetwork()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

}
