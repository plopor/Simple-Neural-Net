import numpy as np

def normalizeSig (x):
    return 1 / (1 + np.exp(-x))

def derivSig (x):
    return x * (1 - x)

trainingIn = np.array([[0, 0, 1],
                       [1, 1, 1],
                       [1, 0, 1],
                       [0, 1, 1]])

learnRate = 0.5

trainingOut = np.array([[0, 1, 1, 0]]).T
#print (trainingOut)
#print (trainingOut.T)

np.random.seed(20)

synapseWeights = 2 * np.random.random((3, 1)) - 1

#print(synapseWeights)

for x in range(20000):
    inputLayer = trainingIn
    outputs = normalizeSig(np.dot(inputLayer, synapseWeights))
    error = -2*(trainingOut - outputs)
    adjust = error * derivSig(outputs)
    synapseWeights -= learnRate * (np.dot(inputLayer.T, adjust))

#print (adjust)

#print (error)
    
#print (synapseWeights)

print (outputs)
print ("vs.")
print (trainingOut)
