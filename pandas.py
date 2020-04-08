# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:22:27 2020

@author: ldelaguila
"""

#%%
import pandas as pd
import numpy as np
print(pd.__version__)
print(pd.show_versions(as_json=True))


#%% Leer y Guardar
df = pd.read_csv("prueba.csv", parse_dates=True, index_col='Date')
df.to_csv("gapminder.csv")
df.to_excel('file_clean.xlsx', index=False)

#%% Crear DataFrame
ser = pd.Series(np.random.randint(1, 10, 35))
df = pd.DataFrame(ser.values.reshape(7,5))

ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser1.name="alphabets" #renombrar serie
ser2 = pd.Series(np.arange(26))
df = pd.concat([ser1, ser2], axis=1)
df = pd.DataFrame({"col1": ser1, 'col2': ser2})


#%%Pasar a DataFrame
mylist = list('abcedfghijklmnopqrstuvwxyz') #crear lista
myarr = np.arange(26) #crear array
mydict = dict(zip(mylist, myarr)) #crear diccionario
ser = pd.Series(mydict) #crear serie
df=ser.to_frame().reset_index() #pasar serie "ser" a data frame

#%% Operaciones con DataFrames
#de ser1, quitarles los de ser2
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
ser1[~ser1.isin(ser2)] 
ser_u = pd.Series(np.union1d(ser1, ser2))  # union
ser_i = pd.Series(np.intersect1d(ser1, ser2))  # intersect
ser_u[~ser_u.isin(ser_i)] #filas que no son comunes


#%% EJERCICIOS

#%%Pasar a mayusculas la primera letra de una serie
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
ser.map(lambda x: x.title())
ser.map(lambda x: x[0].upper() + x[1:])
pd.Series([i.title() for i in ser])

#%%calcular numero caracteres
ser.map(lambda x: len(x))

#%%calcular diferencias
ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
print(ser.diff().tolist())
print(ser.diff().diff().tolist())


#%%Sacar dias del mes, semanas, etc
ser_ts = ser.map(lambda x: parse(x))
print("Date: ", ser_ts.dt.day.tolist())
print("Week number: ", ser_ts.dt.weekofyear.tolist())
print("Day number of year: ", ser_ts.dt.dayofyear.tolist())
print("Day of week: ", ser_ts.dt.weekday_name.tolist())

#%%Modificar fechas
ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])
# Solution 1
from dateutil.parser import parse
ser_ts = ser.map(lambda x: parse(x))
ser_datestr = ser_ts.dt.year.astype('str') + '-' + ser_ts.dt.month.astype('str') + '-' + '04'
[parse(i).strftime('%Y-%m-%d') for i in ser_datestr]
# Solution 2
ser.map(lambda x: parse('04 ' + x))

#%%Palabras con al menos dos volcales
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
from collections import Counter
mask = ser.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiou')]) >= 2)
ser[mask]

#%%Filtrar correos validos
emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@tcohh.hh', 'narendra@modi.com'])
# Solution 1 (as series of strings)
import re
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+//.[A-Za-z]{2,4}'
pattern2=('egypt')
mask = emails.map(lambda x: bool(re.match(pattern2, x)))
emails[mask]
# Solution 2 (as series of list)
emails.str.findall(pattern, flags=re.IGNORECASE)
# Solution 3 (as list)
[x[0] for x in [re.findall(pattern, email) for email in emails] if len(x) > 0]



#%% Mean of a series grouped by another series
fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))
print(weights.tolist())
print(fruit.tolist())

import dfply as dplyr
df=pd.DataFrame({'weights':weights, 'fruits':fruit})
df >> dplyr.group_by(X.fruits) >> dplyr.summarize(mean=X.weights.mean())
weights.groupby(fruit).mean()

#%% Local maxima/minima in a series

ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])

# Solution
dd = np.diff(np.sign(np.diff(ser)))
peak_locs = np.where(dd == -2)[0] + 1

#%% Fill Nas
df = pd.DataFrame({"A":[None, 1, 2, 3, None, None],  
                   "B":[11, 5, None, None, None, 8], 
                   "C":[None, 5, 10, 11, None, 8]}) 
df.bfill(axis ='rows') 
df.ffill(axis ='rows') 
df.bfill(axis ='columns') 
df.ffill(axis ='columns') 

#%% Import CSV, cada 50 filas
# Solution 1: Use chunks and for-loop
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)
df2 = pd.DataFrame()
for chunk in df:
    df2 = df2.append(chunk.iloc[0,:])


# Solution 2: Use chunks and list comprehension
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)
df2 = pd.concat([chunk.iloc[0] for chunk in df], axis=1)
df2 = df2.transpose()

#%% Importar solo algunas columnas

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', usecols=['crim', 'medv'])
print(df.head())

#%% remove rows in a dataframe present in another dataframe

df1 = pd.DataFrame({'fruit': ['apple', 'orange', 'banana'] * 3,
                    'weight': ['high', 'medium', 'low'] * 3,
                    'price': np.arange(9)})

df2 = pd.DataFrame({'fruit': ['apple', 'orange', 'pine'] * 2,
                    'weight': ['high', 'medium'] * 3,
                    'price': np.arange(6)})


# filas en comun
print(df1[df1.isin(df2).all(1)])

# filas no en comun
print(df1[~df1.isin(df2).all(1)])


#%% 
# Input
df = pd.DataFrame(np.random.randint(1, 10, 20).reshape(-1, 4), columns = list('abcd'))

# Solution
pd.value_counts(df.values.ravel())











