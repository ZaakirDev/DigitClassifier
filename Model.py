import numpy as np
import pandas as pd

LEARNING_RATE = 0.01 
EPOCHS = 20

data_train = pd.read_csv("./DataFiles/mnist_train.csv")
data_train = np.array(data_train)

y_train = data_train[:, 0]  # Labels
X_train = data_train[:, 1:]  # Features (pixel values)

# Normalize pixel values from 0-255 to 0-1
X_train = X_train / 255.0

data_test = pd.read_csv("./DataFiles/mnist_test.csv")
data_test = np.array(data_test)

y_test = data_test[:, 0]  # Labels
X_test = data_test[:, 1:]  # Features (pixel values)

# Normalize pixel values
X_test = X_test / 255.0
data_train.shape
def reLU(z):
    return np.maximum(0, z)

def softMax(outLayer):
    return np.exp(outLayer) / np.sum(np.exp(outLayer))

def forwardProp(i, w1, b1, w2, b2):
    z1 = w1 @ i + b1
    a1 = reLU(z1)
    
    z2 = w2 @ a1 + b2
    a2 = softMax(z2)

    return z1, a1, z2, a2

def initializeParams(i):
    # Layer 1 (Hidden Layer)
    w1 = np.random.randn(32, len(i)) * 0.01 
    b1 = np.zeros(32) 

    # Layer 2 (Output Layer)
    w2 = np.random.randn(10, 32) * 0.01 
    b2 = np.zeros(10) 

    return w1, b1, w2, b2

def lossFunction(P, Q):
    return -np.sum(P * np.log(Q))

def reLU_derivative(z):
    return z > 0

def oneHot(P):
    array = np.zeros(10)
    array[P] = 1
    return array

def backPropogation(Z1, A1, A2, W2, X, P):
    dW2 = np.outer(A2 - P, A1)
    dB2 = (A2 - P)
    
    dZ1 = (W2.T @ (A2 - P)) * reLU_derivative(Z1)
    
    dW1 = np.outer(dZ1, X)
    dB1 = dZ1
    
    return dW1, dB1, dW2, dB2

def updateParameters(dW1, dB1, w1, b1, dW2, dB2, w2, b2, n):
    w1 = w1 - (n * dW1)
    b1 = b1 - (n * dB1)

    w2 = w2 - (n * dW2)
    b2 = b2 - (n * dB2)

    return w1, b1, w2, b2
def gradientDescent(i, w1, b1, w2, b2, P, n, accuracyCount, accuracyTotal, index, lossCount, lossTotal):
    global y_train

    z1, a1, z2, a2 = forwardProp(i, w1, b1, w2, b2)
    
    predictedValue = np.argmax(a2)
    accuracyCountValue = y_train[index]

    lossValue = lossFunction(P, a2)
    lossTotal += lossValue
    lossCount += 1

    if predictedValue == accuracyCountValue:
        accuracyCount += 1
    accuracyTotal += 1

    dW1, dB1, dW2, dB2 = backPropogation(z1, a1, a2, w2, i, P)

    w1, b1, w2, b2 = updateParameters(dW1, dB1, w1, b1, dW2, dB2, w2, b2, n)

    return w1, b1, w2, b2, accuracyCount, accuracyTotal, lossTotal, lossCount

def runModel(i, w1, b1, w2, b2):
    z1, a1, z2, a2 = forwardProp(i, w1, b1, w2, b2)
    predictedValue = np.argmax(a2)

    return predictedValue

# Training Phase
if __name__ == "__main__":
    w1, b1, w2, b2 = initializeParams(X_train[0])

    for j in range(EPOCHS):
        accuracyCount = 0
        accuracyTotal = 0

        lossCount = 0
        lossTotal = 0

        for i in range(0, len(X_train) - 1):
            w1, b1, w2, b2, accuracyCount, accuracyTotal, lossTotal, lossCount = gradientDescent(X_train[i], w1, b1, w2, b2, oneHot(y_train[i]), LEARNING_RATE, accuracyCount, accuracyTotal, i, lossCount, lossTotal)
        
        avgLoss = lossTotal / lossCount
        trainingAccuracy = (accuracyCount / accuracyTotal) * 100

        print(f"Training Accuracy: {trainingAccuracy:.2f}%")
        print(f"Average Loss: {avgLoss:.2f}")

    np.savez('./DataFiles/WeightsAndBiases.npz', w1=w1, b1=b1, w2=w2, b2=b2)
