package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Neural network structure with up to 4 hidden layers
type NeuralNetwork struct {
	inputNodes          int           // INPUT NODES (USER ADDED)
	hiddenLayers        int           // HIDDEN LAYERS (USER ADDED)
	hiddenNodesPerLayer []int         // HIDDEN NODES per [hiddenLayer] (USER ADDED)
	hiddenWeights       [][][]float64 // - weights per [hiddenLayer][hiddenNode][inputNode]
	hiddenBias          [][]float64   // - bias per [hiddenLayer][hiddenNode]
	outputNodes         int           // OUTPUT NODES (USER ADDED)
	outputWeights       [][]float64   // - weights per [outputNode][hiddenNode]
	outputBias          []float64     // - bias per [outputNode]
	learningRate        float64       // LEARNING RATE (USER ADDED)
}

type trainingIO struct {
	i []float64
	z []float64
}

// Create the MLP neural network
func CreateNeuralNetwork(inputNodes int, hiddenLayers int, hiddenNodesPerLayer []int, outputNodes int, learningRate float64) *NeuralNetwork {

	// Initialize hiddenWeights slice
	// hiddenWeights[hiddenLayers#][hiddenNodesPerLayer#][inputNodesNumber or hiddenNodesPerLayerNumber]
	hiddenWeights := make([][][]float64, hiddenLayers)
	for i := range hiddenWeights {
		hiddenWeights[i] = make([][]float64, hiddenNodesPerLayer[i])
		for j := range hiddenWeights[i] {
			if i == 0 {
				hiddenWeights[i][j] = make([]float64, inputNodes)
			} else {
				hiddenWeights[i][j] = make([]float64, hiddenNodesPerLayer[i-1])
			}
		}
	}

	// Initialize hiddenBias slice
	// hiddenBias[hiddenLayersNumber][hiddenNodesPerLayerNumber]
	hiddenBias := make([][]float64, hiddenLayers)
	for i := range hiddenBias {
		hiddenBias[i] = make([]float64, hiddenNodesPerLayer[i])
	}

	// Initialize outputWeights slice
	// outputWeights[outputNodesNumber][hiddenNodesPerLayerNumber of last hidden layer]
	outputWeights := make([][]float64, outputNodes)
	for i := range outputWeights {
		outputWeights[i] = make([]float64, hiddenNodesPerLayer[hiddenLayers-1])
	}

	// Initialize outputBias slice
	// outputBias[outputNodesNumber]
	outputBias := make([]float64, outputNodes)

	nn := &NeuralNetwork{
		inputNodes:          inputNodes,          // USER PROVIDED
		hiddenLayers:        hiddenLayers,        // USER PROVIDED
		hiddenNodesPerLayer: hiddenNodesPerLayer, // USER PROVIDED
		hiddenWeights:       hiddenWeights,       // - created here
		hiddenBias:          hiddenBias,          // - created here
		outputNodes:         outputNodes,         // USER PROVIDED
		outputWeights:       outputWeights,       // - created here
		outputBias:          outputBias,          // - created here
		learningRate:        learningRate,        // UER PROVIDED
	}

	return nn
}

// Initialize the neural network (weights and bias) from values -1 to 1
func InitializeNeuralNetwork(nn *NeuralNetwork) *NeuralNetwork {

	// Random number generator from 0-1
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// INIT HIDDEN LAYER(s)
	// l is the hidden layer number
	for l := 0; l < nn.hiddenLayers; l++ {
		// h is the hidden node number
		for hn := 0; hn < nn.hiddenNodesPerLayer[l]; hn++ {
			// in is the input/hidden node number
			if l == 0 {
				for in := 0; in < nn.inputNodes; in++ {
					nn.hiddenWeights[l][hn][in] = r.Float64()*2 - 1
				}
			} else {
				for in := 0; in < nn.hiddenNodesPerLayer[l-1]; in++ {
					nn.hiddenWeights[l][hn][in] = r.Float64()*2 - 1
				}
			}
			nn.hiddenBias[l][hn] = r.Float64()*2 - 1
		}
	}

	// INIT OUTPUT LAYER
	// o is the output node number
	for on := 0; on < nn.outputNodes; on++ {
		// n is the hidden node number
		for hn := 0; hn < nn.hiddenNodesPerLayer[nn.hiddenLayers-1]; hn++ {
			nn.outputWeights[on][hn] = r.Float64()*2 - 1
		}
		nn.outputBias[on] = r.Float64()*2 - 1
	}

	return nn
}

// Print the neural network
func printNeuralNetwork(nn *NeuralNetwork) {

	// NOTE: i is the node number
	fmt.Println("Input nodes:", nn.inputNodes)
	fmt.Println("Hidden layers:", nn.hiddenLayers)

	// Print weights and bias for each hidden layer
	for l := 0; l < nn.hiddenLayers; l++ {
		fmt.Println("HIDDEN LAYER", l, "NODES:", nn.hiddenNodesPerLayer[l])

		// Print weights for each nodes in this hidden layer
		for hn := 0; hn < nn.hiddenNodesPerLayer[l]; hn++ {
			fmt.Println("    weights node:", hn, nn.hiddenWeights[l][hn])
		}
		// Print bias for each node in this hidden layer
		for hn := 0; hn < nn.hiddenNodesPerLayer[l]; hn++ {
			fmt.Println("    bias node:   ", hn, nn.hiddenBias[l][hn])
		}
	}

	// Print the output layer
	fmt.Println("OUTPUT NODES:", nn.outputNodes)
	// Print weights for each nodes in the output layer
	for on := 0; on < nn.outputNodes; on++ {
		fmt.Println("    weights node:", on, nn.outputWeights[on])
	}
	// Print bias for each node in the output layer
	for on := 0; on < nn.outputNodes; on++ {
		fmt.Println("    bias node:   ", on, nn.outputBias[on])
	}

	// Print the learning rate
	fmt.Println("Learning rate    ", nn.learningRate)

}

// Print training data
func printTrainingData(trainingData []trainingIO) {

	fmt.Println("Training Data:")
	for i := 0; i < len(trainingData); i++ {
		fmt.Println("    i ", trainingData[i].i, " z ", trainingData[i].z)
	}

}

// Normalize input data to the range [0, 1] based on the min and max values
func normalize(input []float64) []float64 {

	normalized := make([]float64, len(input))
	minVal := input[0]
	maxVal := input[0]
	// Find the minimum and maximum values in the input data
	for _, v := range input {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	// Normalize the input data to the range [0, 1]
	for i, v := range input {
		normalized[i] = (v - minVal) / (maxVal - minVal)
	}
	return normalized

}

// STEP 1: FORWARD PASS - SUMMATION AND ACTIVATION FUNCTIONS
func forwardPass(nn *NeuralNetwork, x []float64) (yOutput []float64, yHidden [][]float64) {

	// Initialize the hidden outputs for each layer
	yHidden = make([][]float64, nn.hiddenLayers)
	for l := 0; l < nn.hiddenLayers; l++ {
		yHidden[l] = make([]float64, nn.hiddenNodesPerLayer[l])
	}

	// Initialize the output outputs
	yOutput = make([]float64, nn.outputNodes)

	// Calculate the output of each hidden node for each layer
	for l := 0; l < nn.hiddenLayers; l++ {
		for hn := 0; hn < nn.hiddenNodesPerLayer[l]; hn++ {
			yHidden[l][hn] = 0.0
			s := 0.0
			// SUMMATION FUNCTION
			if l == 0 {
				for in := 0; in < nn.inputNodes; in++ {
					s += x[in] * nn.hiddenWeights[l][hn][in]
				}
				s += nn.hiddenBias[l][hn]
			} else {
				for in := 0; in < nn.hiddenNodesPerLayer[l-1]; in++ {
					s += yHidden[l-1][in] * nn.hiddenWeights[l][hn][in]
				}
				s += nn.hiddenBias[l][hn]
			}
			// ACTIVATION FUNCTION
			yHidden[l][hn] = sigmoid(s)
		}
	}

	// Calculate the output of each output node
	for o := 0; o < nn.outputNodes; o++ {
		yOutput[o] = 0.0
		s := 0.0
		// SUMMATION FUNCTION
		for hn := 0; hn < nn.hiddenNodesPerLayer[nn.hiddenLayers-1]; hn++ {
			s += yHidden[nn.hiddenLayers-1][hn] * nn.outputWeights[o][hn]
		}
		s += nn.outputBias[o]
		// ACTIVATION FUNCTION
		yOutput[o] = sigmoid(s)
	}

	return yOutput, yHidden
}

// Activation functions
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// Backpropagate the delta (error)
func backPass(nn *NeuralNetwork, x []float64, yHidden1 [][]float64, yOutput []float64, z []float64) (deltaOutput []float64, deltaHidden [][]float64) {


    // Initialize the hidden deltas for each layer
	deltaHidden = make([][]float64, nn.hiddenLayers)
	for l := 0; l < nn.hiddenLayers; l++ {
		deltaHidden[l] = make([]float64, nn.hiddenNodesPerLayer[l])
	}

    // Initialize the output deltas
    deltaOutput = make([]float64, nn.outputNodes)

	// Calculate the delta for each output node
	for on := 0; on < nn.outputNodes; on++ {
		deltaOutput[on] = z[on] - yOutput[on]
	}

	// Calculate the delta of each hidden node on each layer
    // Starting with the last layer first
	for l := nn.hiddenLayers-1; l < 0; l-- {
		for hn := 0; hn < nn.hiddenNodesPerLayer[l]; hn++ {
            if l == nn.hiddenLayers-1 {
                // On last hidden layer, Use output layer delta and output layer weights
    			deltaHidden[l][hn] += deltaOutput[0] * nn.outputWeights[0][hn]
            } else {
                // Use hidden delta and hidden weights
                deltaHidden[l][hn] = 4
            }
        }
    }

	// Calculate the delta of each hidden node for each layer
    // Starting at the last layer
    for hn := 0; hn < nn.hiddenNodesPerLayer[nn.hiddenLayers-1]; hn++ {
        error := 0.0
        for o := 0; o < nn.outputNodes; o++ {
            error += deltaOutput[o] * nn.outputWeights[o][h]
        }
        deltaHidden[nn.hiddenLayers-1][h] = error * sigmoidDerivative(yHidden1[nn.hiddenLayers-1][h])
    }

	for h := 0; h < nn.hiddenNodesPerLayer[]; h++ {
		error := 0.0
		for o := 0; o < nn.outputNodes; o++ {
			error += deltaOutput[o] * nn.outputWeights[o][h]
		}
		deltaHidden1[h] = error * sigmoidDerivative(yHidden1[h])
	}





	// Update weights and bias for the output layer
	for o := 0; o < nn.outputNodes; o++ {
		for h := 0; h < nn.hidden1Nodes; h++ {
			nn.outputWeights[o][h] += nn.learningRate * deltaOutput[o] * yHidden1[h]
		}
		nn.outputBias[o] += nn.learningRate * deltaOutput[o]
	}

	// Update weights and bias for the hidden layer
	for h := 0; h < nn.hidden1Nodes; h++ {
		for i := 0; i < nn.inputNodes; i++ {
			nn.hiddenWeights1[h][i] += nn.learningRate * deltaHidden1[h] * x[i]
		}
		nn.hiddenBias1[h] += nn.learningRate * deltaHidden1[h]
	}

	return deltaOutput, deltaHidden1
}

func main() {

	// Neural network structure
	inputNodes := 3
	hiddenLayers := 1
	hiddenNodesPerLayer := []int{4}
	outputNodes := 2
	learningRate := 0.1
	epochs := 2

	// Create a new neural network
	fmt.Println("CREATE NEURAL NETWORK --------------------------------------")
	nn := CreateNeuralNetwork(inputNodes, hiddenLayers, hiddenNodesPerLayer, outputNodes, learningRate)

	// Initialize the neural network (weights and bios) with numbers from -1 to 1
	fmt.Println("INITIALIZE NEURAL NETWORK ----------------------------------")
	nn = InitializeNeuralNetwork(nn)

	// Print the neural network
	fmt.Println("PRINT NEURAL NETWORK ---------------------------------------")
	printNeuralNetwork(nn)

	// TrainingData
	trainingData := []trainingIO{

		{i: []float64{85.0, 100.0, 90.1}, z: []float64{1, 90}},
		{i: []float64{60.0, 50.0, 70.0}, z: []float64{0, 70}},
		//{i: []float64{75.0, 80.0, 80.0}, z: []float64{1, 80}},
		//{i: []float64{50.0, 20.0, 55.0}, z: []float64{0, 60}},
		//{i: []float64{90.0, 110.0, 95.0}, z: []float64{1, 95}},
		//{i: []float64{40.0, 30.0, 45.0}, z: []float64{0, 50}},
		//{i: []float64{70.0, 60.0, 75.0}, z: []float64{1, 85}},
		//{i: []float64{55.0, 40.0, 50.0}, z: []float64{0, 60}},
	}

	// Print training InputData
	fmt.Println("PRINT TRAINING DATA --------------------------------------")
	printTrainingData(trainingData)

	// Train the neural network
	fmt.Println("TRAIN NEURAL NETWORK -------------------------------------")
	for epoch := 0; epoch < epochs; epoch++ {
		for _, data := range trainingData {

			// Print the input training data
			fmt.Println("")
			fmt.Println("Input Training Data i: ", data.i)

			// Normalize the input data (0 to 1)
			xInput := normalize(data.i)

			// Print the normalized input data
			fmt.Println("Normalized Input x:    ", xInput)

			// Forward pass - Calculate the output
			fmt.Println("STEP 1 - FORWARD PASS ************************************")
			yOutput, yHidden := forwardPass(nn, xInput)

			// Print the hidden layer and output layer values
			for l := 0; l < hiddenLayers; l++ {
				fmt.Println("   Output Hidden Layer", l, ":", yHidden[l])
			}
			fmt.Println("   Output:         ", yOutput)

			// Calculate the delta (error) and Backpropagate
			fmt.Println("STEP 2 - BACKPASS ****************************************")
			deltaOutput, deltaHidden := backPass(nn, x, yHidden, yOutput, data.z)

			// Print the deltas
			fmt.Println("    Delta Output: ", deltaOutput)
			for l := 0; l < hiddenLayers; l++ {
				fmt.Println("    Delta Hidden Layer", l, ":", deltaHidden[l])
			}

		}
	}
}
