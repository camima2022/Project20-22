from tensorflow.keras import layers
from tensorflow.keras.models import Model

input = layers.Input(shape=(784,))
x = layers.Embedding(input_dim= 1000,output_dim= 128,input_length=784)(input)
model = Model(inputs = input , outputs = x)
model.summary()