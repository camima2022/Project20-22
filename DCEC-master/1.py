import numpy as np
epochs = 20
sum = 0
for epoch in range(epochs):
    if int(epochs/10) == 0 or epoch % int(epochs/4) == 0:
        print(epoch)
        sum += 0