package main

import (
	"fmt"
	"perceptron-simple-example/neuron"
)

// PUBLIC - Configuration
type PerceptronConfiguration struct {
	n int // number of input nodes
}

// PRIVATE - Perceptron
type perceptron struct {
	x []float64      // 1 of more input nodes
	o *neuron.Neuron // 1 output node
}

// Make Perceptron
func (p PerceptronConfiguration) makePerceptron() *perceptron {

	// Build
	in := make([]float64, p.n)
	on := neuron.MakeNeuron()

	// Assemble
	per := perceptron{
		x: in,
		o: on,
	}

	// RETURN
	return &per
}

// RUN
func (p *perceptron) predict(input []float64) {

	p.x[0] = input[0]
	p.x[1] = input[1]

	p.o.B = 1

}

// Print current state of Perceptron
func (p *perceptron) printPerceptron() {

	fmt.Println("INPUT NODES")
	for i, in := range p.x {
		fmt.Printf("  x[%d] %v\n", i, in)
	}
	fmt.Println("OUTPUT NODE")
	fmt.Printf("  o    %v\n", p.o.B)

}

func main() {

	// Configuration
	cfg := PerceptronConfiguration{
		n: 2,
	}

	// Make perceptron with default values
	p := cfg.makePerceptron()

	// Print Perceptron
	p.printPerceptron()

	// predict
	input := []float64{1, 0}
	p.predict(input)

	// Print Perceptron
	p.printPerceptron()

}
