# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:26:31 2018

@author: shkim
"""
import numpy as np

#class Affine:
#    def __init__(self, W, b):
#        self.W = W
#        self.b = b
#        self.x = None
#        self.dW = None
#        self.db = None
#        
#    def forward(self, x):
#        self.x = x
#        out = np.dot(x, self.W) + self.b
#        return out
#    
#    def backward(self, dout):
#        dx = np.dot(dout, self.W.T)
#        self.dW = np.dot(self.X.T, dout)
#        self.db = np.sum(dout, axis=0)
#        return dx
		
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        #print('fp_x shape: ', x.shape)
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):        
        dx = np.dot(dout, self.W.T)
        #print('bp_npdot_dx:', dx.shape)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        #print('bp_reshape_dx:', dx.shape)
        return dx
      
#%%
X = np.random.rand(2)  
W = np.random.rand(2,3)  
B = np.random.rand(3)    

print(X.shape)    # (2,)
print(W.shape)    # (2,3)
print(B.shape)    # (3,) 

Y = np.dot(X, W) + B
print(Y.shape)    # (3,)

#%%
X = np.array([0, 10])
W = np.array([[0, 0, 0], [1, 1, 1]])
print(X.shape, W.shape)    # (2,) (2, 3)
X_dot_W = np.dot(X, W)
print(X_dot_W)    # [10 10 10]
B = np.array([1, 2, 3])
Y = np.dot(X, W) + B
print(Y)    # [11 12 13]

X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])
Y = X_dot_W + B
print(Y)    # [[ 1  2  3] [11 12 13]]

#%%
# dY(N,3) : N(2) -> dB(3,) : sum(dY)
dY = np.array([[1, 2, 3], [4, 5, 6]])
dB = np.sum(dY, axis=0)
print(dB)    # [5 7 9]
dB = np.sum(dY, axis=1)
print(dB)    # [6, 15]

#%%
X = np.array([[0], [10]])     # 2x1
W = np.array([[0, 1], [0, 1], [0, 1]])   # 3x2
print(X.shape, W.shape)    # (2,1) (3,2)
W_dot_X = np.dot(W, X)
print(W_dot_X)    # [[10] [10] [10]]

#%%
X = np.array([[0], [10]])     # 2x1
W = np.array([[0, 0, 0], [1, 1, 1]])   # 2x3
print(X.shape, W.shape)    # (2,1) (2,)
X_dot_W = np.dot(X.T, W)
print(X_dot_W)    # [[10 10 10]]



#%%
