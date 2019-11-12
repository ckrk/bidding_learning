import tensorflow
import keras
import numpy as np

# Check GPU availability
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras.models import Sequential
from keras.layers import Dense, Activation
    

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

#these parameters work, but numpy is fastest and gpu slowest
N, D_in, H, D_out = 64, 1000, 100, 10


# Create random input and output data
    
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Initialize model, should be equivavalent to the pytorch architecture of Linear->Relu->Linear

model = Sequential()
model.add(Dense(H,input_shape=(D_in,)))
model.add(Activation('relu'))
model.add(Dense(D_out))

# For a mean squared error regression problem
# We use adam optimizer to contrast against the pytorch MWE
model.compile(optimizer='adam',
              loss='mse')

# Train model
model.fit(x,y,N)