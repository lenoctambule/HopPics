from HopfieldNet import *

class HopDb:
    def __init__(self, data_len):
        self.hopnet = HopfieldNet(((data_len * 2**8))/ 0.14)

    def tobinary(self, str):
        res = ''.join([ format(ord(i), '08b') for i in str ])
        return [ 1 if i == '1' else -1 for i in res ]

    def tostr(b):
        str_bin = ''.join([ ])
        
    def store(self, data):
        self.hopnet.train([self.tobinary(data)])

    def predict(self, data):
        self.hopnet.predict()