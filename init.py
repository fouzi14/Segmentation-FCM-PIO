from PIL import Image
import numpy as np
import random

w = 100
h = 100

data = np.zeros((h,w,3) , dtype=np.uint8)



for i in range(20):
    for j in range(100):
        data[i, j] = [20 ]

for i in range(20,40):
    for j in range(100):
        data[i, j] = [70 ]

for i in range(40,60):
    for j in range(100):
        data[i, j] = [125 ]

for i in range(60,80):
    for j in range(100):
        data[i, j] = [200 ]

for i in range(80,100):
    for j in range(100):
        data[i, j] = [254 ]
image = Image.fromarray(data)
image.save('save\Imagex.png')
image.show()