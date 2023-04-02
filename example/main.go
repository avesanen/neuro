package main

import (
	"fmt"

	"github.com/avesanen/neuro"
)

func main() {
	// Create network
	inputs := 3
	hidden1 := 3
	hidden2 := 3
	outputs := 3
	n := neuro.NewNetwork([]int{inputs, hidden1, hidden2, outputs})

	// Teach network, loop 100 times
	for i := 0; i < 100; i++ {
		n.FeedForward([]float64{1.0, 1.0, 0.0})
		n.BackPropagation([]float64{1.0, 1.0, 0.0})
	}

	// Get output
	n.FeedForward([]float64{1.0, 1.0, 0.0})
	results := n.Results()

	// print each float in results array with 2 decimals
	for _, result := range results {
		fmt.Printf("%.2f ", result)
	}

}
