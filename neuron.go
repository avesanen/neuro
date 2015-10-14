package neuro

import (
	"math"
	"math/rand"
)

type Neuron struct {
	Outputval                  float64
	Gradient                   float64
	Inputs                     []*Synapse
	Outputs                    []*Synapse
	TransferFunction           func(float64) float64
	TransferFunctionDerivative func(float64) float64
	Eta                        float64
	Alpha                      float64
}

func NewNeuron() *Neuron {
	n := &Neuron{
		Eta:                        0.2,
		Alpha:                      0.5,
		TransferFunction:           func(v float64) float64 { return 1.0 / (1.0 + math.Exp(-v)) },
		TransferFunctionDerivative: func(v float64) float64 { return v * (1 - v) },
	}
	return n
}

func (n *Neuron) Link(to *Neuron) {
	s := &Synapse{
		Input:  n,
		Output: to,
		Weight: rand.Float64(),
	}
	n.Outputs = append(n.Outputs, s)
	to.Inputs = append(to.Inputs, s)
}

func (n *Neuron) FeedForward() {
	var sum float64
	for _, fn := range n.Inputs {
		sum += fn.Input.Outputval * fn.Weight
	}
	n.Outputval = n.TransferFunction(sum)
}

func (n *Neuron) UpdateInputWeights() {
	for _, nn := range n.Inputs {
		nn.DeltaWeight = n.Eta*nn.Input.Outputval*n.Gradient + n.Alpha*nn.DeltaWeight
		nn.Weight += nn.DeltaWeight
	}
}

func (n *Neuron) CalcOutputGradients(targetVal float64) {
	delta := targetVal - n.Outputval
	n.Gradient = delta * n.TransferFunctionDerivative(n.Outputval)
}

func (n *Neuron) CalcHiddenGradients() {
	n.Gradient = n.SumDOW() * n.Outputval
}

func (n *Neuron) SumDOW() float64 {
	var sum float64
	for i, nn := range n.Outputs {
		if i == len(n.Outputs) {
			break
		}
		sum += nn.Weight * nn.Output.Gradient
	}
	return sum
}
