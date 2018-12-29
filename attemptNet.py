import numpy as np

def normalizeSig (x):
    return 1 / (1 + np.exp(-x))

def derivSig (x):
    return x * (1 - x)

#for every example, select the value from the index the net believes to be the answer
def makeOut (index, trainingIn):
    newOut = []
    for i in range(len(trainingIn)):
        newOut.append(trainingIn[i][index])
    return np.array([newOut]).T

#find the input neuron with the highest weight after training
def guess (synapseWeights):
    currMax = -999999999999999999
    index = -1
    for i in range(len(synapseWeights)):
        if (currMax <= synapseWeights[i][0]):
            currMax = synapseWeights[i][0]
            index = i
    return index

trainingIn = np.array([[0, 0, 2],
                       [2, 3, 2],
                       [2, 0, 2],
                       [0, 3, 2]])

learnRate = 0.5
exampleNum = 4
iteration = 80000

trainingOut = np.array([[0, 3, 0, 3]]).T

print ("Training inputs:")
print (trainingIn)
print ("Training outputs, expected results:")
print (trainingOut)

np.random.seed(20)

synapseWeights = 2 * np.random.random((3, 1)) - 1

print ("Random initial synapse weights:")
print(synapseWeights)

for x in range(iteration):
    inputLayer = trainingIn
    outputs = normalizeSig(np.dot(inputLayer, synapseWeights))
    error = -2*(trainingOut - outputs)
    adjust = error * derivSig(outputs)
    #exampleNum is used here to divide the sum of the weights for each of the input neurons
    #by the amount of examples to find the average weight. Since a learning rate is multiplied here as well,
    #the actual division operation doesn't really matter. It is included as clarity.
    synapseWeights -= learnRate * ((np.dot(inputLayer.T, adjust)) / exampleNum)

print ("New synapse weights after training:")
print (synapseWeights)

#given the new weights, the correct number from the list can be selected
print ("Comparison after training:")
print (makeOut (guess (synapseWeights), trainingIn))
print ("vs.")
print (trainingOut)

while True:
    a = float(input("First number: "))
    b = float(input("Second number: "))
    c = float(input("Third number: "))

    print ("Guess:")
    print ((makeOut (guess (synapseWeights), np.array([[a, b, c]])))[0][0])


