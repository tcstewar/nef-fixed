import numpy as np

import neuron

NEURON_BITS = 12
ENCODER_BITS = 12
VALUE_BITS = 10





class Ensemble:
    def __init__(self, N, D, max_rate=(100,200), intercept=(-0.9,0.9), seed=None):
        self.N = N
        self.D = D
        rng = np.random.RandomState(seed)
        self.rng = rng
        
        # compute desired alpha, j_bias for target max_rate and intercept
        tau_ref = 0.002
        tau_rc = 0.016
        max_rates = rng.uniform(max_rate[0], max_rate[1], N)
        intercepts = rng.uniform(intercept[0], intercept[1], N)
        x = 1.0 / (1 - np.exp(
                (tau_ref - (1.0 / max_rates)) / tau_rc))
        alpha = (1 - x) / (intercepts - 1.0)
        j_bias = 1 - alpha * intercepts
        
        

        # bias
        bias = (j_bias*(1<<NEURON_BITS)).astype(np.int16)
        
        
        # encoders
        samples = rng.randn(N, D)
        norm = np.sqrt(np.sum(samples*samples, axis=1))
        encoder = samples/norm[:,None]        
        encoder = encoder * alpha[:,None]
        
        self.encoder = (encoder * (1<<ENCODER_BITS) ).astype(np.int16).T
        
        self.neurons = neuron.FixedNeuron16(N, bias, fractional_bits=NEURON_BITS)
        
        self.decoders = {}
        
       

    def encode(self, x):
        J = np.dot(np.array(x).astype(np.int32), self.encoder) >> (ENCODER_BITS + VALUE_BITS - NEURON_BITS)
        return J
        
    def compute_activity(self, X, T=200, initT=100):
        A = []
        for x in X:
            J = self.encode(x)
            for i in range(initT):   # run the neuron for a while to reset it
                self.neurons.tick(J)
            spikes = np.zeros(self.N)
            for i in range(T):         # run the neuron and count spikes
                self.neurons.tick(J)
                spikes += self.neurons.spikes
            A.append(spikes)    
        A = np.array(A)
        A = A * 1000.0 / T
        return A
        
    def create_decoder(self, name, func=None, noise=0.1, sample_count=None):
        if sample_count is None:
            sample_count = self.D*100
        if func is None: func = lambda x: x    
        
        samples = self.rng.randn(sample_count, self.D)
        norm = np.sqrt(np.sum(samples*samples, axis=1))
        radius = self.rng.uniform(0,1,sample_count)**(1.0/self.D)
        scale = radius / norm        
        samples = samples*scale[:,None]    
        
        target = func(samples)
        
        X = (samples*(1<<VALUE_BITS)).astype(np.int16)
        FX = (target*(1<<VALUE_BITS)).astype(np.int16)
        
        A = self.compute_activity(X)
        
        A += self.rng.randn(*A.shape)*(noise*np.max(A))
        d = np.linalg.lstsq(A, FX)[0]*1000
        
        
        d = d.astype(np.int16)

        
        self.decoders[name] = d
        
     
    def decode(self, name):
        d = self.decoders[name]
        s = self.neurons.spikes
        
        return np.sum(d.astype(np.int32)[s==1], axis=0)
        

        
        
        
