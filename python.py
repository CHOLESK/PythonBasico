#Cadenas
print("hola")
print("Un texto\tuna tabulación")
print("Un texto\nuna nueva línea")
print(r"C:\nombre\directorio")  # r(cadena) => raw (cruda)
print("""Una línea
otra línea
otra línea\tuna tabulación""")

s = "Una cadena" " compuesta de dos cadenas"
diez_espacios = " " * 10
palabra = "Python"
palabra[0] # carácter en la posición 0
palabra[-1] #caracter ultima posicion
palabra = "Python"
palabra[0:2]
palabra[2:]
palabra[:]
palabra[:99]
palabra = "N" + palabra[1:]  #Da Nython
len(palabra) #Da 6

string="hello, world"
print(string.upper())
print(string.lower())
print(string.title())
print(string.split()) #split en una lista de dos palabras. split('\n') separa por lineas
print("".join(string)) #el espacio del principio es la separacion ,pero puede ser ", " o hasta '\n'
print(string.replace("h", "J"))
print("       Hello World   ".strip()) #tambien se pueden quitar otros caracteres del principio o final, con .strip(!)
print('smooth'.find('t')) #dara 4
def favorite_song_statement(song, artist):
  return "My favorite song is {} by {}.".format(song, artist)



#%%
#Listas-----------------------------------------------------------------------------------------------------
datos = [4, "Una cadena", -15, 3.14, "Otra cadena"]
type(datos)
print(datos[0])
print(datos[-1])
print(datos[2:])
cadena_volteada = cadena[::-1] #Da la vuelta a la cadena
numeros = [1,2,3,4]
numeros + [5,6,7,8]
pares = [0,2,4,5,8,10]
pares[3] = 6
pares.append(12)
pares[:2]
#password = "theycallme"crazy"91"  #esta cadena da error
password = "theycallme\"crazy\"91" #esta no

a = [1,2,3]
a.reverse() #modificar 'a' y cambiarle su orden
suma=sum(a)
b = [4,5,6]
c = [7,8,9]
r = [a,b,c]
print(r[0])       # Primera sublista
print(r[-1])      # Última sublista
print(r[0][0])    # Primera sublista, y de ella, primer ítem
print(r[1][1])    # Segunda sublista, y de ella, segundo ítem
print(r[2][2])    # Tercera sublista, y de ella, tercer ítem
print(r[-1][-1])  # Última sublista, y de ella, último ítem

list1 = range(9)  #crear lista de numeros con range
print(list(list1))

cities = ['London', 'Paris', 'Rome', 'Los Angeles', 'New York']
sorted_cities= sorted(cities, reverse=True)
cities.index('Paris')
cities[0].count('o') #cuantas veces aparece una 'o'
cities.count('Paris') #cuantas veces aparece Paris

number = my_list.pop()  #quita el ultimo elemento de la lista, y lo guarda como number

#%%
#ciclos
for x in range(100): #hacer algo 100 veces

lista=range(4)

for i in lista:
  print(i)
  if i == numero:
    print("They have the dog I want!")
    break
  if i is True:
    print("eh")
    continue


words = ["@coolguy35", "#nofilter", "@kewldawg54", "reply", "timestamp", "@matchamom", "follow", "#updog"]
usernames = []
for word in words:
  if word[0] == '@':
    usernames.append(word) #este codigo es lo mismo que lo siguiente:

#%%
#comprehensions
usernames = [word for word in words if word[0] == '@']
celsius = [0, 10, 15, 32, -5, 27, 3] #Para pasar a fahrenheit
fahrenheit = [temp*(9/5) + 32 for temp in celsius]
matrix = [[col for col in range(5)] for row in range(5)]
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
new_fellowship = [member if len(member) >= 7 else "" for member in fellowship]#da ['', 'samwise', '', 'aragorn', 'legolas', 'boromir', '']
new_fellowship = {member: len(member) for member in fellowship}
new_fellowship = [member if len(member) >= 7 for member in fellowship] #este da error
new_fellowship = [member for member in fellowship if len(member) >= 7]


#%%
#generators
result=(member for member in fellowship)
print(next(result))#y para imprimirlo hace falta

#funcion generadora
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']
def get_lengths(input_list):
    for person in input_list:
        yield(person)
for value in get_lengths(lannister):
    print(value)

#%%
#iteradores
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']
superspeed = iter(flash) #esto es obligatorio crearse. Otro ejemplo googol = iter(range(10))
print(next(superspeed))
print(next(superspeed))
print(next(superspeed))
print(next(superspeed))
flash_list = list(enumerate(flash)) #asigna numeros en orden a los integrantes de la lista
print(flash_list)
for index, value in enumerate(words, start=1):
    print(index, value)

result = (num for num in range(31)) #generador

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))

#%%
#DateTime
import random
random.randint(1,101) #numero aleatorio del 1 al 100
random.sample(range(1000), 12) #12 numeros aleatorios del 1 al 1000

from datetime import datetime

birthday = datetime(1991, 12, 31, 17, 00, 12) #año, mes, dia, hora, min, seg
birthday.year #o month, day, hour, weekday, 
datetime.now()
datetime(2018, 1, 1)-datetime(2017, 12, 12)

print(datetime.strptime("Jan 15, 2018", "%b %d, %Y"))
print(datetime.strftime(datetime.now(), "%b %d, %Y"))

#%%
#diccionarios
my_dict = {}
dictionary = {"living room": 21, "kitchen": 23, "bedroom": 20}
students_in_classes = {"software design": ["Aaron", "Delila", "Samson"], "cartography": ["Christopher", "Juan", "Marco"], "philosophy": ["Frederica", "Manuel"]}
dictionary["new_key"] = "new_value" #actualizar diccionario:añadirlo o sobreescribir valor
del(dictionary["living room"]) #borra ese registro
my_dict.update({"key1": 22, "key2": 25, "key3": 34}) #actualziar tambien
my_dict.get("teraCoder", 100000) #obtener el valor de teraCoder, y si no aparece, que marque 100000
my_dict.pop("teraCoder", 0) #quitar esa key del diccionario. si no aparece, ponerla con valor 0
dictionary.keys()
my_dic.values()
my_dict.items()

biggest_brands = {"Apple": 184, "Google": 141.7, "Microsoft": 80, "Coca-Cola": 69.7, "Amazon": 64.8} #acceder a todos los elementos
for company, value in biggest_brands.items():
  print(company + " has a value of " + str(value) + " billion dollars. ")
print("Apple" in biggest_brands) #prints True
names = ['Jenny', 'Alexus', 'Sam', 'Grace']
heights = [61, 70, 67, 64]
students = {key:value for key, value in zip(names, heights)} #una forma de hacer diccionario
students = dict(zip(names, heights)) #otra forma

europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 }}
print(europe["france"]["capital"])
data = {"capital": "rome","population": 59.83}
europe["italy"] = data

names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
my_dict = {"country":names,"drives_right": dr,"cars_per_cap": cpc} #la tercera forma de hacer un diccionario

z1 = zip(mutants, powers) #mutants, powers son dos vectores

for mutants, powers in z1:
    print(mutants, powers)  #muestra los elementos

print(z1) #esto no me lo imprime, tengo que hacer print(list(z1)
print(*z1) #esto me lo imprime, porque * descomprime
result1, result2=zip(*z1) #esto da que result1=mutants, result2=powers

#%%
#matplotlib
print(gdp_cap[-1]); print(life_exp[-1]) #varios comandos en una linea, con punto y coma

import matplotlib.pyplot as plt

plt.plot(year, pop) #year y pop son listas. Grafica por lineas
plt.show() #siempre se pone al final
plt.clf() #y eso para limpiar la grafica. Poner antes de hacer otra

plt.scatter(gdp_cap, life_exp, s = np_pop) #grafica de puntos. s es size. np_pop es un array, numpy.array(pop)
#♣Otros argumentos son c = col, del tipo diccionario. Un color por pais, por ejemplo
alpha=0.8
df.plot(subplots=True) #subplots para cada columna del data frame, o bien
df.subplots(nrows=2, ncols=1) #preparar los subplots

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

plt.subplot(2, 1, 1) #añadir subplot




#%%
#Leer archivos
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


#%%
#funciones

def funcion(word1, echo = 1): #echo es valor por defecto
    global team #establecer una variable como global
    if blabla
        raise ValueError('echo must be greater than 0')
    try:
        #Por aqui continua el codigo
    except:  
        print("Ha habido un error")
        
        
#unas funciones dentro de otras, ejemplo1
def three_shouts(word1, word2, word3):
    def inner(word):
        return word + '!!!'
    return (inner(word1), inner(word2), inner(word3))
three_shouts("a", "b", "c")

# ejemplo2
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo

twice = echo(2) #twice se define entonces como funcion
print(twice('hello')) #se llama a funcion(hello)

#ejemplo 3
def outer():
    global x
    x = "local"
    def inner():
        nonlocal x #decirle que coja el x de fuera
        x = "nonlocal" #lo modifica a nivel de outer
        print("inner:", x)
    
    inner()
    print("outer:", x)

outer()

def gibberish(*args): #funcion con numero variable de argumentos
    hodgepodge = " "
    for word in args:
        hodgepodge += word
    return hodgepodge
one_word = gibberish("luke")
many_words = gibberish("luke", "leia", "han", "obi", "darth")

def report_status(**kwargs): #para pasar un diccionario, con varios argumentos
    for key, value in kwargs.items():
        print(key + ": " + value)
report_status(name = "luke", affiliation = "jedi", status = "missing")

#LAMBDA
add_bangs = (lambda a: a + '!!!') #funcion en una linea
print(add_bangs('hello'))# se llama asi
# Otro ejemplo
echo_word = (lambda word1, echo: word1*echo)
echo_word("hey", 5)
#si se aplica a varios objetos, se hace con map:
nums = [2, 4, 6, 8, 10]
result = list(map(lambda a: a ** 2, nums))
#si se quiere filtrar, se usa filter
spells = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']
print(list(filter(lambda item: len(item) > 6, spells)))
#y si se quieren evitar los duplicados, se usa reduce
from functools import reduce
product = reduce((lambda x, y: x * y), [1, 2, 3, 4]) #producto de un set de numeros

#crear nueva columna
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

#%%
#RE regular expression

import re

prog = re.compile('\d{3}-\d{3}-\d{4}') #formato de 123-456-7890
result = prog.match('111-222-3333')
print(bool(result)) # da True
print(bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890')))
print(bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45')))
print(bool(re.match(pattern='[A-Z]\w*', string='Australia')))

matches = re.findall('\d+', 'string 2 con numeros 4') #da una lista con 2, 4

assert g1800s.iloc[:, 1:].apply(funcion_cualquiera, axis=1).all().all() #aplicar funcion a cada columna de DF

