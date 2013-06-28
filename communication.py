import ensemble


a = ensemble.Ensemble(N=20, D=1, seed=1)
a.create_decoder('X')

b = ensemble.Ensemble(N=20, D=1, seed=2)
b.create_decoder('X')


import math
def input(t):
    return [int(math.sin(t*0.01)*(1<<10))]
    

decay = math.exp(-1/20.0)

xs = []
ys = []
zs = []
y = 0
z = 0
for t in range(500):
    
    x = input(t)
    
    a.neurons.tick(a.encode(x))
    
    y = decay*y + (1-decay)*a.decode('X')
    
    b.neurons.tick(b.encode(y))

    z = decay*z + (1-decay)*b.decode('X')
    
    
    xs.append(x)
    ys.append(y)
    zs.append(z)
    

import pylab
pylab.plot(xs)
pylab.plot(ys)
pylab.plot(zs)
pylab.show()

    
    
    

