# source("libreria.R")
dir()
#library(libreria)
import os
import csv
import pandas as pd
import numpy as np

# setwd("C:/Users/laguila/Desktop")
os.chdir("C:/Users/ldelaguila/Google Drive/ENRI")
# getwd()
print(os.getcwd())
#%% LECTURA Y ESCRITURA
# read.csv(file = "archivo.csv", header = F, sep = ",", row.names = T, col.names = T, skip = 10, nrows = 1000)
# data.table::fread("archivo.csv", stringsAsFactors = F)
np.loadtxt('digits.csv', delimiter="," , skiprows=1, usecols=[0,2], dtype=str)
recfromcsv(file, delimiter = ",", names = True, dtype = None)
genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
pd.read_csv('titanic.csv', header = None, nrows = 5, comment='#', na_values=['Nothing'], delimiter=" ")
xl = pd.ExcelFile('battledeath.xlsx')
df1 = xl.parse("2004") #nombre del sheet
df2 = xl.parse(0, skiprows=[0], names=['Country', 'AAM due to War (2002)'], parse_cols=[0])#primera hoja, renombrando columnas

#Leer a trozos
df_chunk = pd.read_csv(r'../input/data.csv', chunksize=1000)
chunk_list = []  # append each chunk df here 
for chunk in df_chunk:  #esto ya va leyendo trozo a trozo
    chunk_filter = chunk_preprocessing(chunk)
    chunk_list.append(chunk_filter)
df_concat = pd.concat(chunk_list)


# write.csv(dataframe, "archivo.csv")
fichero.to_csv('nombre.csv')

ls #saber que tengo en mi directorio
import glob
csv_files = glob.glob('*.csv') #lista de archivos csv en mi directorio
import os
wd = os.getcwd()
os.listdir(wd)

file = open("moby_dick.txt", "r")
print(file.read())
file.close()
print(file.closed)# Check whether file is closed

with open("world_dev_ind.csv") as file:
    file.readline() # lee la primera linea. no la imprime
    counts_dict = {}
    for j in range(1000):      
        line = file.readline().split(',')# Split the current line into a list: line
        first_col = line[0]
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1
print(counts_dict)

#%% VECTORES Y MATRICES
# a=c(1, 2, 3)
a=np.array([1,2,3])
# a[1]
a[1]
# a[1:2]
a[1:3]
# runif(n = 4, min = 0, max = 10)
import random
np.random.shuffle(a)
np.random.normal(0, 1, 5)
np.random.rand(8)
np.random.randint(2, 5, 7)
np.random.choice(['apple', 'banana', 'carrot'], 10)
# sample(10)
np.random.sample()
#seq(0,10,length.out = 4)
np.linspace(0,10,4)
# a[1]=abs(-5)
a[0]=abs(-5)

# matriz=matrix(data = 1:100, nrow = 10, byrow=T)
matriz = np.arange(100).reshape(10,10,)
# matriz[1:5,1]=4
matriz[0:4,0]=4

# #Trasponer matrix
# t(matrix)
np.transpose(matriz)
matriz.T
#matriz*matriz
matriz*matriz
#matrix %*% matriz
matriz @ matriz
#%% LISTAS
# lista=list(a, 2*a, 4)
lista=[a, 2*a, 4]
# lista[[1]]
lista[1]

#%% DATAFRAMES

# aux=data.frame(unos=rep(1,10), otros=1:10)
df = pd.DataFrame({"unos": np.repeat(1,10), 'col2': np.arange(10)})
# aux$unos
df["unos"]
df.unos

#%% data(iris)
from sklearn.datasets import load_iris
iris = load_iris()
iris=load_iris()
datos=pd.DataFrame(iris.data)
datos['species']=iris.target
datos.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
datos.dropna(how="all", inplace=True) # remove any empty lines
datos["Species"]=datos["Species"].replace(0, iris.target_names[0])
datos["Species"]=datos["Species"].replace(1, iris.target_names[1])
datos["Species"]=datos["Species"].replace(2, iris.target_names[2])


# aux=data.frame(iris)
iris=datos.copy()


pd.set_option('display.max_columns', len(train_oh))
#%% table(iris$Sepal.Length, iris$Sepal.Width)
from collections import Counter
Counter(datos.Species)
datos.Species.value_counts() #similar al table
#Para verlo en todo un dataframe
pd.value_counts(datos.values.ravel())
# head(iris)
iris.head()
# tail(iris)
iris.tail()
# summary(iris)
iris.describe()
# dim(iris)
iris.shape()
# str(iris)
iris.info()
# sum((iris$Sepal.Length))
iris["SepalLength"].sum()
# min(iris$Sepal.Length)
iris["SepalLength"].min()
# max(iris$Sepal.Length)
iris["SepalLength"].max()
# mean(iris$Sepal.Length)
iris["SepalLength"].mean()
# median(iris$Sepal.Length)
iris["SepalLength"].median()
# sd(iris$Sepal.Length)
iris["SepalLength"].std()
# quantile(x = iris$Sepal.Length, 0.35)
np.quantile(iris["SepalLength"],0.35)
# cor(iris[,1:4])
iris[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].corr(method="pearson")
#unique(iris[,5])
np.unique(iris['Species'])
set(iris['Species'])

ser = pd.Series(np.random.normal(0.5, 0.05, 100)) # stats.norm.rvs(0.5, 0.05, size=100)
print(ser.head())
pd.qcut(x=ser, q=[0, 0.2, 0.5, 0.7, 1], labels=["1", "2", "3", "4"]).value_counts()
np.percentile(ser, q=[0, 25, 50, 75, 100]) #sacar percentiles

#%% which(data$Sepal.Length<6)
np.where(iris["SepalLength"]<6)
# data[1,4]=0.3
iris.iloc[0, 3]=0.3
iris.values[0,3]
iris.iloc[:, :]
iris.values
iris.iloc[:,list(range(0,3))]
iris.values[:,0:4]
iris.loc[0, ['PetalWidth']]=0.3

setosa=iris.loc[iris['Species']=='setosa',:]
mask = iris['Species'] == 'setosa'
mask = iris['Species'] == np.nan
iris.loc[mask,:]

# ver si hay NAs
iris.isnull().values.any()
#cuantos NAs hay en cada columna
iris.apply(lambda x: x.isnull().sum())
#rellenarlos
iris[['SepalLength']]=iris[['SepalLength']].apply(lambda x: x.fillna(x.mean()))
# data[which(data$Sepal.Length<5.2),1]=NA
iris.loc[iris['SepalLength']<5.2,['SepalLength']]=np.NaN
iris.loc[iris.SepalLength<5.2,["SepalLength"]]=np.NaN
# data[is.na(data)]=5.2
iris=iris.fillna(5.2)
# data=na.omit(data)
iris=iris.dropna(how='any').reset_index()
iris.dropna(how='all')
# data$Sepal.Length[str_detect(data$Species, pattern = "set")]="esta"
iris.loc[iris['Species'].str.contains('set'), 'Species']='esta'
# str_replace(data$Species, "set", "sat")
iris["Species"].str.replace("set", "sat")

# variable="Sepal.Length"
# match(variable, colnames(iris))
# colnames(iris)[colnames(iris) %in% variable]
variable='SepalLength'
variable in iris.columns #True
iris.iloc[:,iris.columns==variable]

#%%Libreria dfply
from dfply import *
#https://dfply.readthedocs.io/en/docs/autoapi/dfply/index.html
iris >> select(X.PetalLength, X.PetalWidth)
iris >> drop (X.Species)
iris >> mask (X.Species=="setosa") #filter
iris >> arrange(X.PetalLength, ascending=False)
iris >> mutate (PetalNew=X.PetalLength*X.PetalWidth, Otra=1)
iris >> group_by(X.Species) >> summarize (media=X.PetalLength.mean())
iris["Suma"] = iris.groupby(["Species"])["PetalLength"].transform('sum')
iris["Suma2"] = iris.groupby(["Species"])["PetalLength"].transform(lambda x: sum(x))
iris2= iris >> group_by(X.Species) >> summarize(Suma3=X.PetalLength.sum()) >> full_join(copia, by="Species")

#%%ifelse(y==0, 0, 1)
np.where((iris.SepalLength < 5), 0, 1)

# data=iris %>% 
# distinct(Species, Sepal.Length, .keep_all = T)
iris.drop_duplicates()
iris.drop_duplicates(subset='PetalLength')
# #ordenando
# data=iris %>% 
# arrange(Sepal.Length, Sepal.Width)
iris.sort_values("PetalLength", ascending=False)
# data$ceros=0
iris['ceros']=0
# data$ceros=NULL
del(iris['ceros'])
iris.pop("ceros")

#%% data=data.frame(x1=rep(1,10), x2=rep(2, 10))
data = pd.DataFrame({'x1':np.ones(10), 'x2':np.repeat(2, 10)}) #np.full(10, -1)
# data2=data.frame(x3=rep(3,10), x4=rep(4, 10))
data2 = pd.DataFrame({'x3':np.repeat(3, 10), 'x4':np.repeat(4, 10)})
# data3=data.frame(x1=rep(1,10), x5=rep(5, 10))
data3 = pd.DataFrame({'x1':np.ones(10), 'x5':np.repeat(5, 10)})
# total=left_join(data, data3, by="x1")
# total2=merge(data, data2, by="x1")
data.merge(data3, on='x1', how='left')
pd.merge(data, data3, on='x1', how='left')
pd.concat([data, data3], keys=['x', 'y']) #el x e y son para diferenciarlos
pd.concat({'x':data, 'y':data3})
# total3=cbind(data, data2)
total3=pd.concat([data, data3], axis=1)
# total4=rbind(data, data2)
total4=pd.concat([data, data3], axis=0)
total4=data.append(data3)

V = [1,2,3,4,5,6 ]
Y = [7,8,9,10,11,12]
W=np.r_[V[0:2],Y[0],V[3],Y[1:3],V[4:],Y[4:]]
X=np.c_[V, Y]
np.r_[V, Y, W]
np.c_[X, V, Y]
X.ravel()
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)

#%% data=spread(data, key=Species, value=Sepal.Length)
#Las especies en columnas, y el valor el de SepalLength. index las que no se tocan
spread=pd.pivot_table(iris, values='SepalLength', index=['SepalWidth', 'PetalLength', 'PetalWidth'], columns='Species').reset_index()
# data=gather(data, key=Species, value=Sepal.Length, 4:6)
#columnas de species en una columna Species, y SepalLEngth el valor
iris2=spread.melt(id_vars=['SepalWidth', 'PetalLength', 'PetalWidth'], var_name='Species', value_name='SepalLength').dropna(how='any').reset_index()


#%% #renombrar filas y columnas
# colnames(iris)=c("n1", "n2", "n3", "n4")
# rownames(iris)=1:nrow(iris)
iris.columns=['SapalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
iris.rename(columns={'Species': 'Especies'},
            index={0:'cero',1:'uno'})

# or
iris.columns.values[4] = "Especies"

# Step 2:
df.columns = df.columns.map(lambda x: x.replace('.', '_'))
print(df.columns)
#%% #Formatear columnas
# iris$Sepal.Length=as.integer(iris$Sepal.Length)
iris.SepalLength.map(lambda x: int(x))
iris.SepalLength = iris.SepalLength.astype('int32')
iris = iris.astype('int32') #toda las variables numericas
# iris$Sepal.Length=as.numeric(iris$Sepal.Length)
iris.SepalLength.map(lambda x: float(x))
# iris$Species=as.character(iris$Species)
iris.Species.map(lambda x: str(x))
# iris$Species=as.factor(iris$Species)

for lab, row in cars.iterrows(): #crear columna nueva
    cars.loc[lab, "COUNTRY"] = row["country"].upper()
cars["COUNTRY"] = cars["country"].apply(str.upper) #aunque tambien se puede hacer asi

red_vs_blue = {'Obama':'blue', 'Romney':'red'} 
election['winner'].map(red_vs_blue) #en la columna winner, cambiar Obama por blue, y Romney por red

tips['total_bill'] = pd.to_numeric(tips["total_bill"], errors="coerce") #pasar a numerico
tips.sex = tips.sex.astype("category") 

#%%
import math
# iris$Petal.Length=round(iris$Petal.Length,2)
round(iris['PetalLength'], 1)
iris['PetalLength'].round(1)
# iris$Petal.Length=ceiling(iris$Petal.Length,2)
iris['PetalLength'].map(lambda x: math.ceil(x))
# iris$Petal.Length=floor(iris$Petal.Length,2)
iris['PetalLength'].map(lambda x: math.floor(x))

# rm(data)
del(data)

#%% str_detect(as.character(iris$Species), string = "setosa")
mask = iris['Species'].map(lambda x: bool(re.match('set', x)))
iris['Species'][mask] #columa Spcies
mask_inverse = ~mask #si mask es lista con Trues, lo convierte en False
# substr(as.character(iris$Species), 1, 4)
iris['Species'].map(lambda x: x[0:4])
# str_replace(as.character(iris$Species), pattern = "a", replacement = "e")
iris['Species'].map(lambda x: x.replace('set', 'sat'))
# str_remove(as.character(iris$Species), pattern = "a")
iris['Species'].map(lambda x: x.replace('set', ''))
# colSums(iris[,1:4])
iris.iloc[:,0:4].sum(axis=1)
# rowSums(iris[,1:4])
iris.iloc[:,0:4].sum(axis=1)

#%%Transformar DataFrames
cars.reindex(australianos, method="ffill", how='linear') #bfill
cars['country'].str.contains('A') #filas de los paises que contienen A
cars['country'].str.upper() #Pasar nombres de paises a mayusculas
cars['country'].str.strip() #Quitar espacios en blanco
cars['Date'].dt.hour #Devuelve las horas de los DateTimes
cars['Date'].dt.tz_localize('US/Central')


#Separar strings
column_labels='primera, segunda, tercera'
column_labels_list = column_labels.split(",")

#%% #Crear directorios
# dir.create("carpeta_guardado")
os.mkdir('dir1')
os.makedirs('dir1/dir2/dir3', exist_ok=True) #hacer varios a la vez, y no de error en caso de que exista
# dir.exists("carpeta_guardado")
os.path.isdir('Favorites')

#%%DateTime

my_datetimes = pd.to_datetime(date_list, format='%Y-%m-%d %H:%M')
time_series = pd.Series(temperature_list, index=my_datetimes)  

ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']
ts2 = ts0.loc['2010-07-04']
ts3 = ts0.loc['2010-12-15':'2010-12-31']
ts3 = ts2.reindex(ts1.index)

#Parsear
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
from dateutil.parser import parse
ser.map(lambda x: parse(x))
pd.to_datetime(ser)

#%%Un poco de todo
iris.SepalLength.rolling(2).sum()
#%%Imput Data
population.resample('A').first()interpolate('linear') #Imputar valores anualmente linealmente

#%% MATPLOTLIB
import matplotib.pyplot as plt

plt.plot(df['Mes'], df['data science'], label='data science')
plt.plot(df['Mes'], df['machine learning'], label='machine learning')
plt.plot(df['Mes'], df['deep learning'], label='deep learning')
plt.xlabel('Date')
plt.ylabel('Popularity')
plt.title('Popularity of AI terms by date')
plt.grid(True)
plt.legend()
plt.text(x='2010-01-01', y=80, s=r'$\lambda=1, r^2=0.8$') #Coordinates use the same units as the graph
plt.annotate('Notice something?', xy=('2014-01-01', 30), xytext=('2006-01-01', 50), arrowprops={'facecolor':'red', 'shrink':0.05})

fig, axes = plt.subplots(2,2)
axes[0, 0].hist(df['data science'])
axes[0, 1].scatter(df['Mes'], df['data science'])
axes[1, 0].plot(df['Mes'], df['machine learning'])
axes[1, 1].plot(df['Mes'], df['deep learning'])

plt.scatter(df['data science'], df['machine learning'])

plt.bar(df['Mes'], df['machine learning'], width=20)

plt.hist(df['deep learning'], bins=15)


plt.show() #siempre se pone al final
plt.clf() #y eso para limpiar la grafica. Poner antes de hacer otra



column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
df[column_list2].plot() #plotear solo algunas columnas, guardadas como una lista en column_list2
plt.xscale('log') #eje x logaritmico
plt.xlabel('xlab')
plt.ylabel('ylab')
plt.title('title')
tick_val = [1000, 10000, 100000] #ticks donde se pone el tick lab
tick_lab = ['1k', '10k', '100k']
plt.xticks(tick_val, tick_lab)
plt.xlim(20, 55)
plt.ylim(20, 55)
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')
plt.grid(True)
plt.hist(life_exp, bins = 5) #histogramas
plt.imshow(im_sq, cmap='Greys', interpolation='nearest') #una imagen de 28*28 pixels
plt.show()

df['Existing Zoning Sqft'].plot(kind='scatter', x='Year', y='Total Urban Population') #kind="hist", "box", logx=True, logy=True
df.boxplot(column="initial_cost", by="Borough", rot=90)

