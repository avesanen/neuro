package neuro

import (
	"math"
	"math/rand"
	"time"
)

type Network struct {
	Layers       [][]*Neuron
	Error        float64
	AverageError float64
}

func NewNetwork(topology []int) *Network {
	rand.Seed(time.Now().UnixNano())
	network := &Network{}
	for _, numNeurons := range topology {
		layer := make([]*Neuron, numNeurons)
		for i := range layer {
			layer[i] = NewNeuron()
		}
		network.Layers = append(network.Layers, layer)
	}

	for i := 0; i < len(network.Layers)-1; i++ {
		for _, from := range network.Layers[i] {
			for _, to := range network.Layers[i+1] {
				from.Connect(to)
			}
		}
	}
	return network
}

func (n *Network) Train(inputVals, targetVals []float64) {
	n.FeedForward(inputVals)
	n.BackPropagation(targetVals)
}

func (n *Network) FeedForward(inputVals []float64) {
	inputLayer := n.Layers[0]
	for i, value := range inputVals {
		inputLayer[i].OutputVal = value
	}

	for _, layer := range n.Layers[1:] {
		for _, neuron := range layer {
			neuron.FeedForward()
		}
	}
}

func (n *Network) BackPropagation(targetVals []float64) {
	outputLayer := n.Layers[len(n.Layers)-1]
	n.Error = 0.0

	for i, neuron := range outputLayer {
		delta := targetVals[i] - neuron.OutputVal
		n.Error += delta * delta
	}

	n.Error /= float64(len(outputLayer))
	n.Error = math.Sqrt(n.Error)
	n.AverageError = (n.AverageError + n.Error) / 2.0

	for i, neuron := range outputLayer {
		neuron.CalculateGradients(targetVals[i])
	}

	for i := len(n.Layers) - 2; i > 0; i-- {
		for _, neuron := range n.Layers[i] {
			neuron.CalculateGradients(0)
		}
	}

	for _, layer := range n.Layers[1:] {
		for _, neuron := range layer {
			neuron.UpdateWeights()
		}
	}
}

func (n *Network) Results() []float64 {
	outputLayer := n.Layers[len(n.Layers)-1]
	results := make([]float64, len(outputLayer))
	for i, neuron := range outputLayer {
		results[i] = neuron.OutputVal
	}
	return results
}
