import numpy as np

class HopfieldNet:
    def __init__(self, data_len, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.data_len = data_len
        self.weights = np.zeros((data_len, data_len),dtype=float)

    def train(self,data_arr):
        new = self.weights.copy()
        for data in data_arr:
            for i in range(self.data_len):
                for j in range(self.data_len) :
                    if i != j :
                        new[i,j] += (data[i] * data[j])
        self.weights = np.add(self.weights, new / self.data_len)

    def predict(self,data, steps=2):
        res = data.copy()
        for k in range(steps):
            for i in range(len(data)):
                for j in range(len(data)):
                    if i != j :
                        res[i] += self.weights[i,j] * data[j]
            #print("Step ",k+1,": ",[int(np.sign(i) * -1) for i in res])
        return [int(np.sign(i)) for i in res]
        #return res

import random as rd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    hp = HopfieldNet(data_len=10**2)
    arr = [
        np.zeros((10,10), dtype=int),
        np.zeros((10,10), dtype=int),
            ]
    for i in range(10):
        for j in range(10):
            if j == 4 :
                arr[0][i,j] = 1
            elif i == 0 :
                arr[0][i,j] = 1
            else :
                arr[0][i,j] = -1

    for i in range(10):
        for j in range(10):
            if i == 4:
                arr[1][i,j] = 1
            elif j == i :
                arr[1][i,j] = 1
            else :
                arr[1][i,j] = -1

    train = arr
    hp.train([train[0].flatten(), train[1].flatten()])
    test = [ 1 if rd.randint(0,1) == 0 else -1 for i in range(10**2)]
    #print(hp.weights)
    #print(np.reshape(train, (10,10)))


    res = hp.predict(test, steps=30)



    print(np.reshape(res, (10,10)))
    #print(np.reshape(hp.weights, (100,100)))

    plt.subplot(2,3,1)
    plt.imshow(hp.weights)
    plt.title('Weight matrix')

    plt.subplot(2,3,3)
    plt.imshow(np.reshape([int(np.sign(i)) for i in res], (10,10)))
    plt.title('After')
    
    plt.subplot(2,3,2)
    plt.imshow(np.reshape(test, (10,10)))
    plt.title('Before')

    plt.subplot(2,3,4)
    plt.imshow(arr[0])
    plt.title('Trained pattern 1')

    plt.subplot(2,3,5)
    plt.imshow(arr[1])
    plt.title('Trained pattern 2')
    
    plt.show()

