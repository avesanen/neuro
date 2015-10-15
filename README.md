# neuro
Neural network testing with golang

Install:
```
go get github.com/avesanen/neuro
```

Usage:
```
// Create network
inputs := 3
hidden1 := 3
hidden2 := 3
outputs := 3
n := neuro.NewNetwork([]int{inputs, hidden1, hidden2, outputs)

// Teach network
n.FeedForward([]float64{1.0, 1.0, 0.0})
n.BackPropagate([]float64{0.0, 0.0, 0.0})

// Get output
n.FeedForward([]float64{1.0, 1.0, 0.0})
results := n.Results()
```
