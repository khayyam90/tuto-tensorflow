from os import path
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

        
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()        

x_train = np.reshape(x_train, (60000,28,28,1))
x_test = np.reshape(x_test, (10000,28,28,1))

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = x_train / 255 # Normalize the images to [0, 1]
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
x_test = x_test / 255 # Normalize the images to [0, 1]

latent_dim = (4,4,1) 
input_dim = (28,28,1)

class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_dim),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Flatten(),    
      #tf.keras.layers.Dense(2*np.prod(latent_dim)),
      tf.keras.layers.Dense(3*np.prod(latent_dim)),
      tf.keras.layers.Dense(np.prod(latent_dim)),
      tf.keras.layers.Reshape(latent_dim)
      
    ])
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(3*np.prod(latent_dim)),
      tf.keras.layers.Dense(np.prod(input_dim), activation='sigmoid'),
      tf.keras.layers.Reshape(input_dim)
    ])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
    
autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss=tf.losses.MeanSquaredError())

FILENAME = "encoder.save"
if path.exists(FILENAME):
    autoencoder = tf.keras.models.load_model(FILENAME)
else:

    autoencoder.fit(x_train, x_train,
                    epochs=6,
                    validation_data=(x_test, x_test))
    
    # sauvegarde du réseau
    autoencoder.save(FILENAME)
    
def showDifferences():    
    fig, axs = plt.subplots(10,3)    
    randoms = [2,58,69,74,25,125,324,896,102,478]    
    for i in range(10):
        r = randoms[i]
        x = x_train[r]
        x = x.reshape(1,28,28,1)
        y = autoencoder(x)
        x = x[0]
        y = y[0]
        x = x * 255
        y = y * 255
        diff = np.abs(x-y)
        
        axs[i, 0].imshow(x, cmap=plt.get_cmap('gray'))
        axs[0, 0].set_title("Images originales")
        
        axs[i,1].imshow(y, cmap=plt.get_cmap('gray'))
        axs[0,1].set_title("Images reconstuites")
        
        axs[i,2].imshow(diff, cmap=plt.get_cmap('gray'))
        axs[0,2].set_title("Différences")
    
    plt.show()
#showDifferences()
         
def showAnimation2():
    x = x_train[2587]
    x = x.reshape(1,28,28,1)
    print(x.shape)

    latent = autoencoder.encoder(x)
    latent = np.array(latent)
    print(latent.shape)
    
    plt.imshow(latent[0], cmap=plt.get_cmap("gray"))
    plt.show()
    
    pixel1 = latent[0,1,1,0]
    pixel2 = latent[0,2,2,0]
    
    pixel1Destination = pixel1 - 100
    pixel2Destination = pixel2 + 100
    diff1 = pixel1Destination - pixel1
    diff2 = pixel2Destination - pixel2
    
    os.mkdir("out")
    
    for step in range(40):
        latent[0,1,1,0] = pixel1 + (step/40) * diff1
        latent[0,2,2,0] = pixel2 + (step/40) * diff2
        yhat = autoencoder.decoder(latent)
        yhat = yhat * 255
        plt.imshow(yhat[0])
        plt.savefig("out/out"+str(step)+".png")
           
def morph(item1, item2, i):
    x1 = x_train[item1] # 3
    x1 = x1.reshape(1,28,28,1) # 3

    latent1 = autoencoder.encoder(x1)
    latent1 = np.array(latent1)
    
    x2 = x_train[item2] # 3
    x2 = x2.reshape(1,28,28,1)

    latent2 = autoencoder.encoder(x2)
    latent2 = np.array(latent2)
    
    diff = latent2 - latent1
    
    os.mkdir("out")
    
    for step in range(10):
        latent = latent1 + step/10 * diff
        yhat = autoencoder.decoder(latent)
        yhat = yhat * 255
        plt.imshow(yhat[0])
        plt.savefig("out/"+str(i+step)+".png") 
      
def showMorph():    
    items = [69, 102, 25, 74,  9, 11,   13,    478, 125,      19] 
    imgCount = 0
    for i in range(len(items)):
        morph(items[i],items[(i+1)%len(items)], imgCount)
        imgCount = imgCount+10
        
def denoise():
    x = x_train[5896]
    x = x.reshape(28,28,1)
    
    # add random noise
    noise = np.random.normal(0, .1, x.shape)
    xnoised = x + noise
    
    y = autoencoder([xnoised])
    y = y[0]
    
    fig, axs = plt.subplots(1,2)     
    axs[0].imshow(xnoised, cmap=plt.get_cmap("gray"))
    axs[0].set_title("Image bruitée")
        
    axs[1].imshow(y, cmap=plt.get_cmap("gray"))
    axs[1].set_title("Image reconstuite")
    plt.show()

denoise()    
    