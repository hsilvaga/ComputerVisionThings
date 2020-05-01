import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as pyplot

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0 , test_images / 255.0 #Normalizes data between 0 and 1

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Convolutions extract features from image
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=(32, 32, 3))) #relu = rectify linear
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Classifier
model.add(layers.Flatten()) #creates 1D array of for data
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10)) #10 for number of classes(NEURONS)

model.compile(optimizer = 'adam',
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy'] )

history = model.fit(train_images, train_labels, epochs = 12, validation_data =(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)