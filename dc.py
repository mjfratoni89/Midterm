#data should be named midterm_train.csv

import numpy as np
import csv
import matplotlib.pyplot as plt


data = np.genfromtxt('midterm_train.csv', delimiter=',', skip_header=1)

fig, ax = plt.subplots()

print (data[:,8])

ax.plot(data[:,8])
plt.show()
