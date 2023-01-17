import numpy as np

class HopfieldNet:
    weights = np.zeros((9,9), dtype=float)
    data_len = 9

    def __init__(self, data_len):
        self.data_len = data_len
        self.weights = np.zeros((data_len, data_len),dtype=float)

    def train(self,data_arr):
        new = self.weights.copy()
        for data in data_arr:
            for i in range(self.data_len):
                for j in range(self.data_len) :
                    if i != j :
                        new[i,j] = (data[i] * data[j]) / len(data)
        self.weights = np.add(self.weights, new)

    def predict(self,data, steps=2):
        res = data.copy()
        for k in range(steps):
            for i in range(len(data)):
                for j in range(len(data)):
                    if i != j :
                        res[i] += self.weights[i,j] * data[j]
            #print("Step ",k+1,": ",[int(np.sign(i) * -1) for i in res])
        
        return [int(np.sign(i) * -1) for i in res]


if __name__ == "__main__":
    hp = HopfieldNet(data_len=9)
    train = [
        [-1,-1,-1,-1,1,-1,1,-1, 1],
            ]
    hp.train(train)
    test = [1,1,1,1,-1,1,-1,1,-1]
    #print(hp.weights)
    print("Training pattern :")
    print(np.reshape(train, (3,3)))
    print("Before :")
    print(np.reshape(test, (3,3)))
    test = hp.predict(test, steps=10)
    print("After :")
    print(np.reshape(test, (3,3)))
