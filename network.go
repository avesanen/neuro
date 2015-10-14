package neuro

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Network struct {
	Layers                       [][]*Neuron
	Error                        float64
	RecentAverageError           float64
	RecentAverageSmoothingFactor float64
}

func NewNetwork(topology []int) *Network {
	rand.Seed(time.Now().UnixNano())
	n := &Network{}
	for i, v := range topology {
		var layer []*Neuron
		for nn := 0; nn < v; nn++ {
			layer = append(layer, NewNeuron())
		}

		if i != len(topology)-1 {
			bias := NewNeuron()
			bias.Outputval = 1.0
			layer = append(layer, bias)
		}

		n.Layers = append(n.Layers, layer)
	}

	for layerIndex, _ := range n.Layers {
		if layerIndex == len(n.Layers)-1 {
			break
		}
		for _, from := range n.Layers[layerIndex] {
			for _, to := range n.Layers[layerIndex+1] {
				from.Link(to)
			}
		}
	}
	return n
}

func (n *Network) FeedForward(inputVals []float64) {
	if len(inputVals) != len(n.Layers[0])-1 {
		fmt.Println("ERROR: Network has", len(n.Layers[0])-1, "inputs.")
		return
	}

	for layerIndex, layer := range n.Layers {
		if layerIndex == 0 {
			for i, v := range inputVals {
				layer[i].Outputval = v
			}
		} else {
			for _, neuron := range layer {
				neuron.FeedForward()
			}
		}
	}
}

func (n *Network) BackPropagation(targetVals []float64) {
	// Todo: Check for correct amout of targetVals
	outputlayer := n.Layers[len(n.Layers)-1]
	n.Error = 0.0

	for neuronIndex, neuron := range outputlayer {
		delta := targetVals[neuronIndex] - neuron.Outputval
		n.Error += delta * delta
	}

	n.Error /= float64(len(outputlayer) - 1)
	n.Error = math.Sqrt(n.Error)

	n.RecentAverageError = (n.RecentAverageError + n.RecentAverageSmoothingFactor + n.Error)
	n.RecentAverageError /= n.RecentAverageSmoothingFactor + 1.0

	for neuronIndex, neuron := range outputlayer {
		if neuronIndex == len(outputlayer) {
			break
		}
		neuron.CalcOutputGradients(targetVals[neuronIndex])
	}

	for layerIndex := len(n.Layers) - 2; layerIndex > 0; layerIndex-- {
		for _, neuron := range n.Layers[layerIndex] {
			neuron.CalcHiddenGradients()
		}
	}

	for layerIndex := len(n.Layers) - 1; layerIndex > 0; layerIndex-- {
		for neuronIndex, neuron := range n.Layers[layerIndex] {
			if neuronIndex == len(n.Layers[layerIndex]) {
				break
			}
			//if rand.Intn(10) > 5 {
			neuron.UpdateInputWeights()
			//}
		}
	}
}

func (n *Network) Results() []float64 {
	var resultVals []float64
	for _, neuron := range n.Layers[len(n.Layers)-1] {
		resultVals = append(resultVals, neuron.Outputval)
	}
	return resultVals
}
