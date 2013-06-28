import numpy as np

import neuron

import ensemble

def create_ensemble(N, D, seed=None):
    rng = np.random.RandomState(seed)

    NEURON_BITS = 12
    
    # bias
    bias_range = 1<<NEURON_BITS
    bias = rng.randint(-bias_range, bias_range, N)
    
    # gain
    gain = rng.uniform(0.5, 4, N)**2

    # encoders
    samples = rng.randn(N, D)
    norm = np.sqrt(np.sum(samples*samples, axis=1))
    encoder = samples/norm[:,None]
    
    encoder = encoder * gain[:,None]
    
    encoder = (encoder * (1<<NEURON_BITS) ).astype(np.int16)
    
    n = neuron.FixedNeuron16(N, bias)
    
    return ensemble.Ensemble(n, encoder, decoders={})
    


if __name__=='__main__':
    a = create_ensemble(N=4, D=1, seed=1)
    
    print a.neurons.bias
    print a.encoder
    
