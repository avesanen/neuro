package neuro

import (
	"math"
	"math/rand"
)

type Neuron struct {
	OutputVal                  float64
	Gradient                   float64
	Connections                []*Synapse
	TransferFunction           func(float64) float64
	TransferFunctionDerivative func(float64) float64
	Eta                        float64
	Alpha                      float64
}

func NewNeuron() *Neuron {
	return &Neuron{
		Eta:                        0.2,
		Alpha:                      0.5,
		TransferFunction:           func(v float64) float64 { return 1.0 / (1.0 + math.Exp(-v)) },
		TransferFunctionDerivative: func(v float64) float64 { return v * (1 - v) },
	}
}

func (n *Neuron) Connect(to *Neuron) {
	connection := &Synapse{
		Input:  n,
		Output: to,
		Weight: rand.Float64(),
	}
	n.Connections = append(n.Connections, connection)
	to.Connections = append(to.Connections, connection)
}

func (n *Neuron) FeedForward() {
	sum := 0.0
	for _, conn := range n.Connections {
		sum += conn.Input.OutputVal * conn.Weight
	}
	n.OutputVal = n.TransferFunction(sum)
}

func (n *Neuron) UpdateWeights() {
	for _, conn := range n.Connections {
		conn.DeltaWeight = n.Eta*conn.Input.OutputVal*n.Gradient + n.Alpha*conn.DeltaWeight
		conn.Weight += conn.DeltaWeight
	}
}

func (n *Neuron) CalculateGradients(targetVal float64) {
	delta := targetVal - n.OutputVal
	n.Gradient = delta * n.TransferFunctionDerivative(n.OutputVal)
}
