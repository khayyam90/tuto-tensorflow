import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])
    
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

# once the training is over, let's analyse manual images
files = ["zero.png", "one.png", "two.png", "three.png", "four.png", "five.png", "six.png", "seven.png"]
for file in files:
    img = load_img("in/" + file, color_mode='grayscale')
    x = img_to_array(img)
    x = x.reshape(1,28,28)
    x = x / 255.0
       
    y = model.predict(x)
    y = y[0]
    bestClass = y.argmax()
    print(file + " => " + str(bestClass) + "\n")