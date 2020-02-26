# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:27:34 2020

@author: ldelaguila
"""

import numpy as np
arr=np.arange(10)

np.full((3,3),True, dtype=bool)

arr[arr%2==1]=-1
arr2=np.where(arr%2==1, -1, arr) #ifelse en R
arr=arr.reshape(2, -1) #filas, columnas, pero si se pone -1 decide el num de columnas
arr2=arr2.reshape(2, -1)
#stack vertically
np.concatenate([arr, arr2], axis=1)
# Method 2:
np.vstack([arr, arr2])
# Method 3:
np.r_[arr, arr2]

a = np.array([1,2,3])
np.concatenate([np.repeat(a, 3), np.tile(a, 3)])

#valores en comun
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)
np.setdiff1d(a,b)
np.where(a == b) #posiciones donde los elementos coinciden
a[(a >= 2) & (a <= 4)]

arr = np.arange(9).reshape(3,3)
arr[:, [1,0,2]] #intercambiar columnas 0 y 1

arr[:, ::-1] 

np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))
np.random.uniform(5,10, size=(5,3))
np.set_printoptions(precision=3) #Limitar numero de decimales, en siguientes lineas


