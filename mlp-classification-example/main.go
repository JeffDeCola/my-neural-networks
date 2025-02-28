package main

import (
	"fmt"

	mlp "my-go-packages/mlp"
)

func main() {

	// Neural Network Parameters
	nnp := mlp.NeuralNetworkParameters{
		InputNodes:          2,
		InputNodeLabels:     []string{"midterm-grade", "hours-studied", "last-test-grade"},
		HiddenLayers:        1,
		HiddenNodesPerLayer: []int{2},
		OutputNodes:         1,
		OutputNodeLabels:    []string{"predicted-percentage-passing-final", "predicted-final-grade"},
		LearningRate:        0.1,
		Epochs:              2,
		DatasetCSVFile:      "dataset.csv",
	}

	// Create a new neural network
	fmt.Println("\nCREATE NEURAL NETWORK --------------------------------------")
	nn := nnp.CreateNeuralNetwork()

	// Initialize the neural network (weights and bios with numbers from -1 to 1)
	fmt.Println("\nINITIALIZE NEURAL NETWORK ----------------------------------")
	nn.InitializeNeuralNetwork()

	// Print the neural network
	fmt.Println("\nPRINT NEURAL NETWORK ---------------------------------------")
	nn.PrintNeuralNetwork()

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

	// Train the neural network with dataset from CSV file
	fmt.Println("\nTRAIN NEURAL NETWORK ---------------------------------------")
	nn.TrainNeuralNetwork()

	// Print the trained neural network
	// fmt.Println("\nPRINT NEURAL NETWORK ---------------------------------------")
	// nn.PrintNeuralNetwork()

	//

}
