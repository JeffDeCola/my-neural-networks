// This package will model a neuron
//
//	n will be total number of input (starting at 1)
//	xn will be the input
//	wn will be weights on those inputs
//	b will be bias
//	s will be output of summation function
//	f(s) will be activation function
//	y will be output of activation function

package neuron

type Neuron struct {
	B float64
}

type ActivationFunction func(float64) float64

func MakeNeuron() *Neuron {

	// BUILD
	var bias float64

	// ASSEMBLE
	ner := Neuron{
		B: bias,
	}

	// RETURN
	return &ner
}
