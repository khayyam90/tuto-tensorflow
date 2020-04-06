import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#y_train = tf.keras.utils.to_categorical(y_train, 10)

print(y_train[0])

#y_test = tf.keras.utils.to_categorical(y_test, 10)
#y_train = tf.keras.utils.to_categorical(y_train, 10)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(loss=loss_fn,
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          epochs=2,
          verbose=1,
)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)


# once the training is over, let's analyse manual images
files = ["zero.png", "one.png", "two.png", "three.png", "four.png", "five.png", "six.png", "seven.png"]
for file in files:
    img = load_img("in/" + file, color_mode='grayscale')
    x = img_to_array(img)
    x = x.reshape(1,28,28,1)
    x = x / 255.0
       
    y = model.predict(x)
    y = y[0]
    bestClass = y.argmax()
    print(file + " => " + str(bestClass) + "\n")