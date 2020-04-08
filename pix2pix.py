# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:37:24 2020

@author: laguila
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

#Ruta raiz
PATH="C:/Users/laguila/Desktop/flores"

#Datos de entrada
INPATH = PATH + "/original"

#Datos de salida
OUPATH = PATH + "/blurred"

#CheckPoints
CKPATH = PATH + "/checkpoints"

#Obtener URLS de las imagenes
os.chdir(INPATH)
imgurls=os.listdir()

#Listado randomizado
randurls = np.copy(imgurls)
np.random.seed(23)
np.random.shuffle(randurls)

#Particion train-test
n=500
train_n = round(n*0.8)
tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]

print(len(imgurls), len(tr_urls), len(ts_urls))

#%%

IMG_WIDTH = 256
IMG_HEIGHT = 256

#Reescalar imagenes
def resize(inimg, tgimg, height, width):
    
    inimg = tf.image.resize(inimg, [height, width])
    tgimg = tf.image.resize(tgimg, [height, width])
    
    return inimg, tgimg

#Normalizar al ranto [-1, 1] 
def normalize(inimg, tgimg):
    
    inimg = (inimg / 127.5) - 1
    tgimg = (tgimg / 127.5) - 1
    
    return inimg, tgimg

# Recortar imagen a una zona interna y hacerle flip aleatorio
@tf.function()
def random_jitter(inimg, tgimg):
    
    inimg, tgimg = resize(inimg, tgimg, 286, 286)
    
    stacked_image = tf.stack([inimg, tgimg], axis=0)
    #recorte aleatorio, las dos imagenes, ancho y alto, y 3 canales de color
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    
    inimg, tgimg = cropped_image[0], cropped_image[1]
    
    if np.random.random(1)[0] > 0.5:
        
        #print("ecole")
        inimg = tf.image.flip_left_right(inimg)
        tgimg = tf.image.flip_left_right(tgimg)
        
    return inimg, tgimg

#%%Cargar imagenes

def load_image(filename, augment=True):
    
    inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + '/' + filename)), tf.float32)[...,:3]
    tgimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUPATH + '/' + filename)), tf.float32)[...,:3]

    inimg, tgimg = resize(inimg, tgimg, IMG_HEIGHT, IMG_WIDTH)
    
    if augment:
        inimg, tgimg = random_jitter(inimg, tgimg)
        
    inimg, tgimg = normalize(inimg, tgimg)
    
    return inimg, tgimg


def load_train_image(filename):
    return load_image(filename, True)

def load_test_image(filename):
    return load_image(filename, False)


plt.imshow(((load_train_image(randurls[80])[0])+1)/2)

#%% 

train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
train_dataset = train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(1)

for inimg, tgimg in train_dataset.take(5):
    print(tgimg.shape)
    print(type(tgimg))
    plt.imshow((inimg[0,...] + 1)/2)
    plt.show()

test_dataset = tf.data.Dataset.from_tensor_slices(ts_urls)
test_dataset = test_dataset.map(load_test_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(1)

#%%

from tensorflow.keras.layers import *
from tensorflow.keras import *


def downsample(filters, apply_batchnorm = True):
    
    result = Sequential()
    
    #Inicializador media 0, desviacion 0.02
    initializer = tf.random_normal_initializer(0, 0.02) 
    
    #Capa convolucional
    result.add(Conv2D(filters,
                      kernel_initializer = initializer,
                      strides = 2,
                      padding = "same",
                      kernel_size = 4,
                      use_bias = not apply_batchnorm))
    
    #Capa BatchNormalization
    result.add(BatchNormalization())
    
    #Capa de activacion
    result.add(LeakyReLU())
    
    return result


def upsample(filters, apply_dropout = False):
    
    result = Sequential()
    
    #Inicializador media 0, desviacion 0.02
    initializer = tf.random_normal_initializer(0, 0.02) 
    
    #Capa convolucional
    result.add(Conv2DTranspose(filters,
                               kernel_initializer = initializer,
                               strides = 2,
                               padding = "same",
                               kernel_size = 4,
                               use_bias = False))
    
    #Capa BatchNormalization
    result.add(BatchNormalization())
    
    if apply_dropout:
        
        #Añadir capa dropout
        result.add(Dropout(0.5))
        
    
    #Capa de activacion
    result.add(ReLU())
    
    return result


#%% 

def Generator():
    
    #Capa de entrada. None es para no especificar ningunas dimensiones (alto y ancho)
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    
    #Comentarios son dimensiones de imagenes. bs= batchsize, numero de imagenes.
    #El ultimo numero son los filtros
    down_stack = [
        downsample(64, apply_batchnorm = False),    # (bs, 128, 128, 64)
        downsample(128),                            # (bs, 64,  64,  128)
        downsample(256),                            # (bs, 32,  32,  256)
        downsample(512),                            # (bs, 16,  16,  512)
        downsample(512),                            # (bs, 8,   8,   512)
        downsample(512),                            # (bs, 4,   4,   512)
        downsample(512),                            # (bs, 2,   2,   512)
        downsample(512),]                           # (bs, 1,   1,   512)


    up_stack = [
        upsample(512, apply_dropout = True),      # (bs, 2,  2,  1024)
        upsample(512, apply_dropout = True),      # (bs, 4,  4,  1024)
        upsample(512, apply_dropout = True),      # (bs, 8,  8,  1024)
        upsample(512),                            # (bs, 16, 16, 1024)
        upsample(256),                            # (bs, 4,  4,  512)
        upsample(128),                            # (bs, 2,  2,  256)
        upsample(64),]                            # (bs, 1,  1,  128)
    
    #Creacion de la ultima capa, que crea la imagen
    
    initializer = tf.random_normal_initializer(0, 0.02) 
    
    last = Conv2DTranspose(filters = 3, #canales de color
                           kernel_size = 4,
                           strides = 2, #duplicar tamaño imagen
                           padding = "same",
                           kernel_initializer = initializer,
                           activation = 'tanh')
    
    #Conectar las capas, arquitctura UNET
    x = inputs #entrada inicial
    s = []     #skip connections
    
    concat = Concatenate()
    
    for down in down_stack:
        x = down(x)
        s.append(x)
        
    s = reversed(s[:-1])
    
    for up, sk in zip(up_stack, s):
        
        x = up(x)
        x = concat([x, sk])
        
    last = last(x)
    
    return Model(inputs = inputs, outputs = last)
        

#Comprobar que funciona
generator = Generator()

# gen_output = generator(((inimg+1)*255),training = False)
# plt.imshow(gen_output[0,...])
# plt.imshow((tgimg[0,...] + 1)/2)[0,...]

#%% Discriminador PatchGan

def Discriminator():
    
    #dos entradas, la imagen original, y la que genera el generador
    
    ini = Input(shape = [None, None, 3], name = "input_img")
    gen = Input(shape = [None, None, 3], name = "gener_img")
    
    con = concatenate([ini, gen])
    
    initializer = tf.random_uniform_initializer(0, 0.02)
    
    down1 = downsample(64, apply_batchnorm = False)(con)
    down2 = downsample(128)(down1)
    down3 = downsample(256)(down2)
    down4 = downsample(512)(down3)
    
    last = Conv2DTranspose(filters = 1,
                           kernel_size = 4,
                           strides = 1, 
                           padding = "same",
                           kernel_initializer = initializer)(down4)
    
    return tf.keras.Model(inputs = [ini, gen], outputs = last)

discriminator = Discriminator()
# disc_out = discriminator([((inimg+1)*255),gen_output], training = False)
# plt.imshow(disc_out[0, ..., -1], vmin = -20, vmax = 20, cmap = "RdBu_r")
# plt.colorbar()
# disc_out.shape

#%% Funcion de coste adversaria

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def discriminator_loss(disc_real_output, disc_generated_output):
    
    #Diferencia entre los true por ser real y el detectado por el discriminador
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    
    #Diferencia entre los false por ser generado y el detectado por el discriminador
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_real_output)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss


LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    #mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    total_gen_loss = gan_loss + (LAMBDA + l1_loss)
    
    return total_gen_loss


#%%
import os

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)


checkpoint_prefix = os.path.join(CKPATH, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = generator,
                                 discriminator = discriminator)

#Para restaurar por el ultimo checkpoint
#checkpoint.restore(tf.train.latest_checkpoint(CKPATH)).assert_consumed()


#%%

def generate_images(model, test_input, tar, save_filename = False, display_imgs = True):
    
    prediction = model(test_input, training = True)
    
    if save_filename:
        tf.keras.preprocessing.image.save_img(PATH + '/output/' + save_filename + '.jpg', prediction[0,...])
    
    plt.figure(figsize = (10, 10))
    
    display_list = [test_input[0], tar[0], prediction[0]]
    
    title = ['Input Image' 'Ground Truth', 'Predicted Image']
    
    if display_imgs:
        for i in range(3):
            plt.subplot(1, 3, i+1)
            #plt.title(title[i])
            
            #getting pixel values between [0,1] to plot it
            plt.imshow( display_list[i] * 0.5 + 0.5)
            plt.axis('off')
    plt.show()
    
#%% 

@tf.function
def train_step(input_image, target):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
    
        output_image = generator(input_image, training = True)
        
        output_gen_discr = discriminator([output_image, input_image], training = True)
        
        output_trg_discr = discriminator([target, input_image], training = True)
        
        discr_loss = discriminator_loss(output_trg_discr, output_gen_discr)
        
        gen_loss = generator_loss(output_gen_discr, output_image, target)
        
        
        generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)

        discriminator_grads = discr_tape.gradient(discr_loss, discriminator.trainable_variables)

    
        generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
        
        discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))
        
        


    
#%%


from IPython.display import clear_output

def train(dataset, epochs):
    
    for epoch in range(epochs):
        
        imgi = 0
        
        for input_image, target in dataset:
            
            print('epoch' + str(epoch) + ' - train: ' + str(imgi)+'/'+str(len(tr_urls)))
            imgi+=1
            train_step(input_image, target)
            clear_output(wait = True)
            
        imgi = 0
        for inp, tar in test_dataset.take(3):
            generate_images(generator, inp, tar, str(imgi) + ' ' + str(epoch), display_imgs = True)
            imgi+=1
        #saving chechpoints
        
        if(epoch + 1) % 20 == 0:
            
            checkpoint.save(file_prefix = checkpoint_prefix)


#%%

train(train_dataset, 500)









































