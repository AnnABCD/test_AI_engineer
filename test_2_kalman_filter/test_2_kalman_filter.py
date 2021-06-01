from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


################################# PROBLEME 1 #################################

data = pd.read_csv('data\\pb1.csv', header = None)
data = data.transpose()

# Positions observees
mesures_positions = data[0].values

# Vecteur observations positions et vitesses
def Z_observations(positions, n):
    Z = np.zeros((2,n))
    Z[0] = mesures_positions
    # Calcul des vitesses observees
    Z[1,0] = 10
    for i in range(n-1):
        Z[1,i+1] = Z[0,i+1] - Z[0,i]
    # Transposition en vecteur temporel (chaque colonne t corespond a la position et la vitesse a l'instant t)
    Z = np.transpose(Z)
    return(Z)

# Equations de predictions
def predictions(x, P, F, B):
        x_pred = np.dot(F, x)
        P_pred = np.dot(np.dot(F, P), np.transpose(F)) + B
        return x_pred, P_pred
 
# Equations de mises a jour selon l'observation
def update(x_pred, P_pred, z, H, R) :
        y = z - np.dot(H, x_pred)
        S = np.dot(H, np.dot(P_pred, np.transpose(H))) + R
        K = np.dot(np.dot(P_pred, np.transpose(H)), np.linalg.inv(S))
        x_update = x_pred + np.dot(K, y)
        P_update = np.dot(Id - np.dot(K,H), P_pred)
        return x_update, P_update


n = len(mesures_positions)
Z = Z_observations(mesures_positions, n)

#Conditions intiales
X = np.zeros((n,2))
X[0] = [1,1]
Id = np.identity(2)
F = np.array([[1, 1], [0, 1]])
P = 1000*Id
H = Id
B = np.array([[0.5, 0.5], [0.5, 0.5]])
R = np.array([[0.5, 0.5], [0.5, 0.5]])

# Filtre de Kalman
for t in range(n-1):
    X[t+1], P = predictions(X[t], P, F, B)
    X[t+1], P = update(X[t], P, Z[t], H, R)

# Visualisation
time=np.arange(n)
X = np.transpose(X)
Z = np.transpose(Z)

plt.plot(time, Z[0], label = 'Positions mesurees')
plt.plot(time, X[0], label = 'Positions estimees')
plt.legend()
plt.show()


