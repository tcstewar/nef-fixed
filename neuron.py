import numpy as np

# TODO: voltage[i] should be unsigned: going below zero should stay at zero, and
#  going above threshold means firing, so we if we can just detect an overflow
#  we don't have to represent numbers outside [0, 1<<fractional_bits)


class FixedNeuron16:
    def __init__(self, neuron_count, bias, fractional_bits=12):
        self.voltage = np.array([0]*neuron_count, dtype=np.int16)
        self.refractory = np.array([0]*neuron_count, dtype=np.dtype('i1'))
        self.spikes = np.array([0]*neuron_count, dtype=np.dtype('b'))
        self.bias = bias
        self.fractional_bits = fractional_bits

        self.neuron_count = neuron_count

    def tick(self, current):
        decay_shift = 4    # tau_rc = 2**decay_shift
        tau_ref = 2        # milliseconds for refractory period
    
        dv = (current + self.bias - self.voltage) >> decay_shift

        for i in range(self.neuron_count):
            if self.refractory[i]==0:
                self.voltage[i] += dv[i]
                if self.voltage[i]<0: self.voltage[i]=0
            else:
                self.refractory[i] -= 1

            if self.voltage[i] >= 1<<self.fractional_bits:
                self.spikes[i] = 1
                self.refractory[i] = tau_ref
                self.voltage[i] -= 1<<self.fractional_bits
            else:
                self.spikes[i] = 0                

    # alternate array-based implementation (identical functionality to the above code)
    def tick(self, current):
        decay_shift = 4    # tau_rc = 2**decay_shift
        tau_ref = 2        # milliseconds for refractory period
        threshold = 1<<self.fractional_bits
    
        dv = (current + self.bias - self.voltage) >> decay_shift
        
        self.voltage = np.where(self.refractory == 0, np.maximum(self.voltage + dv, 0), self.voltage)
        self.refractory = np.where(self.refractory == 0, 0, self.refractory-1)
        self.spikes = np.where(self.voltage >= threshold, 1, 0)
        assert len(self.spikes.shape)==1
        self.refractory = np.where(self.spikes, tau_ref, self.refractory)
        self.voltage = np.where(self.spikes, self.voltage - threshold, self.voltage)
        

                
                
if __name__=='__main__':
    n = FixedNeuron16(4)
    current = [1<<12, 2<<12, 3<<12, 4<<12]
    s = np.zeros(4)
    for i in range(1000):
        n.tick(current)
        s += n.spikes
        
    print s    
    
                    
