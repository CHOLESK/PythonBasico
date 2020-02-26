# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:22:27 2020

@author: ldelaguila
"""

#%%
import pandas as pd
print(pd.__version__)
print(pd.show_versions(as_json=True))

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

#%% Leer y Guardar
df = pd.read_csv("prueba.csv", parse_dates=True, index_col='Date')
df.to_csv("gapminder.csv")
df.to_excel('file_clean.xlsx', index=False)

#%% Visual Exploratory Data Analysis
iris.plot(x='sepal_length', y='sepal_width', kind='scatter', s="petal_length", style='k.-') #k negro, blue, green, red, cyan. o circle, * start, s square, + plus, - dashed
plt.show()

iris.plot(y='sepal_width', kind='area')
plt.show()

iris.plot(y='sepal_width', kind='box')
plt.show()

iris.plot(kind='box', subplots=True)
plt.show()


iris.plot(y='sepal_width', kind='hist', bins=10, range=(4,8), normed=True)
plt.show()

iris.plot(y='sepal_width', kind='hist', bins=10, range=(4,8), normed=True, cumulative=True)
plt.show()

#%%Statistical Exploratory Analysis
cars.shape
cars.columns
cars.info()
cars.describe()
#Lo siguiente se puede aplicar a un DataFrame o a una columna
cars.count()
cars["primera"].count() #number of entries
cars["primera"].mean()
cars["primera"].std()
cars["primera"].min()
cars["primera"].max()
cars["primera"].median()
cars["primera"].quantile(0.5)
cars["primera"].quantile([0.25, 0.75])
cars["primera"].unique()
cars.corr() 

#%%DateTime

my_datetimes = pd.to_datetime(date_list, format='%Y-%m-%d %H:%M')
time_series = pd.Series(temperature_list, index=my_datetimes)  

ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']
ts2 = ts0.loc['2010-07-04']
ts3 = ts0.loc['2010-12-15':'2010-12-31']
ts3 = ts2.reindex(ts1.index)

#Se puede hacer group_by + summarise(media=mean(day)), asi
daily_mean=sales.resample('D').mean() #Daily, Time (minute), Hour, Businesday, Week, Month, Quarter, Año.
daily_mean=sales.resample('6H').mean() #Cada 6 horas

unsmoothed = df['Temperature']['2010-Aug-01':'2010-Aug-15'] #columna temperature, filas que cumplen ese DateTime
smoothed = unsmoothed.rolling(window=2).mean() #Medias moviles

#%%Imput Data
population.resample('A').first()interpolate('linear') #Imputar valores anualmente linealmente

#%%
#Acceder partes de un DataFrame
setosa=iris.loc[iris['species']=='setosa']
mask = iris['species'] == 'setosa'
mask = iris['species'] == np.nan


cars.index =['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG'] #nombre filas
cars.index.name="Pais"
cars.columns.name="Pais"
cars['US'].values #una columna es una serie, que viene de un numpy array
cars.columns = ["primera", "segunda"] #nombre columnas dataframe

cars[:5,:-1] #5 primeras columnas de la ultima fila
cars.head(5)
cars.tail()
print(cars["country"]) #muestra la columna de los paises
print(cars[["country"]]) #muestra columna de los paises, incluyendo country
print(cars[["country", "drives_right"]]) #mostrar varias columnas. No se puede con un solo parentesis
print(cars[0:3]) #muestra 3 primeras filas
print(cars.loc[["AUS", "EG"]]) #mostrar las filas que queramos. Se pueden usar : tambien
print(cars.loc[["AUS":"EG":1]]) #y en orden reverso
print(cars.iloc[[2, 3]]) #filas 3 y 4
print(cars.loc[["RU", "MOR"],["country", "drives_right"]]) #primer parentesis nombre filas, segundo nombre columnas
print(cars.loc[:,:"EG"])  #selecciona todas las filas, y todas las columnas hasta EG
#parentesis primero o segundo se pueden sustituir por : y asi mostrar todo
print(df_clean.loc['2011-6-20 8:00:00':'2011-6-20 9:00:00', 'dry']) #muestra esas filas de tiempo de la columna dry
australianos=cars.loc["AUS"]
grandes = cars['mpg'] > 70
cars_grandes=cars[grandes]

#%%Transformar DataFrames
cars.reindex(australianos, method="ffill", how='linear') #bfill
cars['country'].str.contains('A') #filas de los paises que contienen A
cars['country'].str.upper() #Pasar nombres de paises a mayusculas
cars['country'].str.strip() #Quitar espacios en blanco
cars['Date'].dt.hour #Devuelve las horas de los DateTimes
cars['Date'].dt.tz_localize('US/Central')

#gather y spread
visitors_pivot = users.pivot(index='weekday', columns='city', values='visitors') #columns=key, values=value, index columnas que se quedan

#%%Formatear columnas DataFrame
df['Time'] = df['Time'].apply(lambda x:'{:0>4}'.format(x))
df.astype('int32').dtypes #pasar columnas a integer
df.astype({'col1': 'int32'}).dtypes #solo una columna

df_clean['dry'] = pd.to_numeric(df_clean['dry'], errors='coerce')

ser = pd.Series([1, 2], dtype='int32')
ser.astype('int64') #cambiar tipo una serie

#%%Filtrar y recorrer DataFrame
car_maniac = cars[cars["cars_per_cap"] > 500] #filtrar por filas
medium = cars[np.logical_and(cars['cars_per_cap'] > 100, cars['cars_per_cap'] < 500)]
pop=list(zip(cars['cars_per_cap'], cars['price'])) #quedarme con esas columnas

for lab, row in cars.iterrows(): #iterrar sobre 
    print(lab) #nombre filas
    print(row) #datos columnas

for lab, row in cars.iterrows(): #crear columna nueva
    cars.loc[lab, "COUNTRY"] = row["country"].upper()
cars["COUNTRY"] = cars["country"].apply(str.upper) #aunque tambien se puede hacer asi
cars["prueba"] = cars.variable.str[0:3] #nueva columna con 3 primeras letras de la columna variable
cars["COUNTRY"] = cars.country.apply(upper, 2)
cars.apply(to_celsius) #aplicar funcion a un dataframe
red_vs_blue = {'Obama':'blue', 'Romney':'red'} 
election['winner'].map(red_vs_blue) #en la columna winner, cambiar Obama por blue, y Romney por red
df_reader = pd.read_csv('ind_pop.csv', chunksize = 10)
print(next(df_reader))  #lee los primeros 10 registros. si lo vuelvo a ejecutar, lee del 10 al 20

#%%ordenar
pd.melt(file, id_vars=["columna sin alterar1", "Columna2"], var_name="key", value_name="value") #es un gather
cars.pivot_table(file, index=["Month", "Day"], columns="measurement", values="reading")
cars['nueva_col'] = cars.variable.str.split("_")#separar la columna variable en otra columna (columna de listas)
car['type'] = car.nueva_col.str.get(0) #obtener primer elemento de listas de nueva_col

#%%Unir DataFrames
pd.concat([uber1, uber2, uber3]) #rbind. Concatena listas. Para ir formando listas, usar append
pd.concat([uber1, uber2, uber3], axis=1) #cbind
o2o = pd.merge(left=site, right=visited, left_on="name", right_on="site") #merge

gapminder_agg = gapminder.groupby("year")['life_expectancy'].mean() #group by con summarise

tips['total_bill'] = pd.to_numeric(tips["total_bill"], errors="coerce") #pasar a numerico
tips.sex = tips.sex.astype("category") 
assert gapminder.year.dtypes == np.int64 #comprobar que es del tipo int64
mask_inverse = ~mask #si mask es lista con Trues, lo convierte en False
car.drop_duplicates()#quitar filas duplicadas
car.dropna(how='any') #eliminar toda la fila donde haya un NA
car.dropna(how='all') #eliminar toda la fila solo si hay un NA en todas las columnas
df.drop(columnas_sin_interes, axis='columns') #quitar de df las columnas sin interes
airquality['Ozone'] = airquality.fillna(oz_mean) #reemplazar NAs en columna Ozone por oz_mean, que es la media
airquality['Ozone'].all() #True si todos los valores son True
airquality['Ozone'].notnull() #True si los valores no son nulos
print(airquality.notnull().all().all()) #ve si cada registro es nulo o no, lo aplica para cada fila (all) y eso a cada columna (otro all)
assert(airquality.notnull().all().all()) #si True, no devuelve nada

#%% Operaciones con DataFrames
#de ser1, quitarles los de ser2
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
ser1[~ser1.isin(ser2)] 
ser_u = pd.Series(np.union1d(ser1, ser2))  # union
ser_i = pd.Series(np.intersect1d(ser1, ser2))  # intersect
ser_u[~ser_u.isin(ser_i)] #filas que no son comunes

#%% Distribución aleatoria
ser=pd.Series(np.random.normal(100, 10, 20)) #distribucion normal aleatoria
np.percentile(ser, q=[0, 25, 50, 75, 100]) #sacar percentiles

ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30))) #de la primera lista, tomar los objetos de la segunda
ser.value_counts() #similar al table

np.random.RandomState(100) #setseed
ser = pd.Series(np.random.randint(1, 5, [12]))
print(ser.value_counts()) #hacer table
ser[~ser.isin(ser.value_counts().index[:2])] = 0 #modificar valores serie

ser = pd.Series(np.random.normal(0.5, 0.05, 100))
print(ser.head())
pd.qcut(x=ser, q=[0, 0.2, 0.5, 0.7, 1], labels=["1", "2", "3", "4"]).value_counts()

ser = pd.Series(np.random.randint(1, 10, 7))
a=np.argwhere(ser % 3==0)

#%% EJERCICIOS
column_labels='primera, segunda, tercera'
column_labels_list = column_labels.split(",")


#Calcular error cuadratico medio
truth = pd.Series(range(10))
pred = pd.Series(range(10)) + np.random.random(10)
np.mean((truth-pred)**2)

#Pasar a mayusculas la primera letra de una serie
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
ser.map(lambda x: x.title())
ser.map(lambda x: x[0].upper() + x[1:])
pd.Series([i.title() for i in ser])

#calcular numero caracteres
ser.map(lambda x: len(x))

#calcular diferencias
ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
print(ser.diff().tolist())
print(ser.diff().diff().tolist())

#TimeSeries
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
from dateutil.parser import parse
ser.map(lambda x: parse(x))
pd.to_datetime(ser)

#Sacar dias del mes, semanas, etc
ser_ts = ser.map(lambda x: parse(x))
print("Date: ", ser_ts.dt.day.tolist())
print("Week number: ", ser_ts.dt.weekofyear.tolist())
print("Day number of year: ", ser_ts.dt.dayofyear.tolist())
print("Day of week: ", ser_ts.dt.weekday_name.tolist())

#Modificar fechas
ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])
# Solution 1
from dateutil.parser import parse
ser_ts = ser.map(lambda x: parse(x))
ser_datestr = ser_ts.dt.year.astype('str') + '-' + ser_ts.dt.month.astype('str') + '-' + '04'
[parse(i).strftime('%Y-%m-%d') for i in ser_datestr]
# Solution 2
ser.map(lambda x: parse('04 ' + x))

#Palabras con al menos dos volcales
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
from collections import Counter
mask = ser.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiou')]) >= 2)
ser[mask]

#Filtrar correos validos
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







