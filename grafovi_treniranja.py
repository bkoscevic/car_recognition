import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace


f = open("training.log", "r")
data = []
for line in f:
    data.append(line.split(","))
titles = data[0]
data = np.array(data[1:])

acc, loss, val_acc, val_loss = [], [], [], []
for x in data[:,1]:
    acc.append(float(x)*100)
for x in data[:,2]:
    loss.append(float(x))
for x in data[:,3]:
    val_acc.append(float(x)*100)
for x in data[:,4]:
    val_loss.append(float(x))

plt.figure(figsize=(10,15))
plt.title("Gubitak modela po etapama treniranja")
plt.plot(linspace(0, 100, 100), loss, '-')
plt.grid()
plt.savefig('loss.png')

plt.figure(figsize=(10,15))
plt.title("Preciznost modela po etapama treniranja")
plt.plot(linspace(0, 100, 100), acc, '-')
plt.grid()
plt.savefig('acc.png')

plt.figure(figsize=(10,15))
plt.title("Gubitak modela po etapama validacije treniranja")
plt.plot(linspace(0, 100, 100), val_loss, '-')
plt.grid()
plt.savefig('valloss.png')

plt.figure(figsize=(10,15))
plt.title("Preciznost modela po etapama validacije treniranja")
plt.plot(linspace(0, 100, 100), val_acc, '-')
plt.grid()
plt.savefig('valacc.png')