# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:27:34 2020

@author: ldelaguila
"""

import numpy as np

np.set_printoptions(precision=3) #Limitar numero de decimales, en siguientes lineas
np.set_printoptions(threshold=np.nan) #Imprimir todo el array

arr=np.arange(10)
np.full((3,3),True, dtype='int32')
arr[arr%2==1]=-1
arr2=np.where(arr%2==1, -1, arr) #ifelse en R
arr=arr.reshape(2, -1) #filas, columnas, pero si se pone -1 decide el num de columnas
arr2=arr2.reshape(2, -1)
#%%stack vertically
np.concatenate([arr, arr2], axis=1)
# Method 2:
np.vstack([arr, arr2])
# Method 3:
np.r_[arr, arr2]

#%%valores en comun
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)
np.setdiff1d(a,b)
np.where(a == b) #posiciones donde los elementos coinciden
a[(a >= 2) & (a <= 4)]

#Manipular array
arr = np.arange(9).reshape(3,3)
arr[:, [1,0,2]] #intercambiar columnas 0 y 1
arr[:, ::-1] #Cambiar el orden a las columnas. Similar para filas 

#%%
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# Solution 1
np.corrcoef(iris[:, 0], iris[:, 2])[0, 1]

#%% Rellenar NANs
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Solution
iris_2d[np.isnan(iris_2d)] = 0

#%% Numerical to categorical
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')


# Bin petallength 
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])

# Map it to respective category
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]

# View
petal_length_cat[:4]

#%% most frequent value, moda

a=np.random.randint(1, 10, 100)
vals, counts = np.unique(a, return_counts=True)
print(vals[np.argmax(counts)])
print(vals[counts==counts.max()])

#%% Primera vez que sale un 9 o mayor
a=np.random.randint(1, 10, 100)
np.argwhere(a >= 9)[0]

#%% Valores mayores de cada fila o columna
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
a

# Solution 1
np.amax(a, axis=1)




