package neuro

type Synapse struct {
	Weight      float64
	DeltaWeight float64
	Input       *Neuron
	Output      *Neuron
}
