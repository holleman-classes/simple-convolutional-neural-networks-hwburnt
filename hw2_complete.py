### Add lines to import modules as needed
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import Input, layers, Model
from PIL import Image
##

def build_model1():
  model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), strides=(2,2), padding="same", input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv2D(64, (3,3), strides=(2,2), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv2D(128, (3,3), strides=(2,2), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    # Four more Conv2D+BatchNorm layers without striding
    layers.Conv2D(128, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv2D(128, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv2D(128, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv2D(128, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    # Max pooling
    layers.MaxPooling2D(pool_size=(4, 4), strides=(4,4)),

    # Flatten
    layers.Flatten(),

    # Dense layers
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(10, activation="softmax"),
  ])
  return model


def build_model2():
  model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), strides=(2,2), padding="same", input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.SeparableConv2D(64, (3,3), strides=(2,2), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.SeparableConv2D(128, (3,3), strides=(2,2), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    # Four more Conv2D+BatchNorm layers without striding
    layers.SeparableConv2D(128, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.SeparableConv2D(128, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.SeparableConv2D(128, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.SeparableConv2D(128, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    # Max pooling
    layers.MaxPooling2D(pool_size=(4, 4), strides=(4,4)),

    # Flatten
    layers.Flatten(),

    # Dense layers
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(10, activation="softmax"),
  ])
  return model

def build_model3():
  inp = Input(shape=(32,32,3))
  y = layers.Conv2D(32, (3,3), activation = 'relu',strides=(2,2), padding="same")(inp)
  x = layers.BatchNormalization()(y)
  x = layers.Dropout(0.3)(x)

  x = layers.Conv2D(64, (3,3), strides=(2,2), activation = 'relu', padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Conv2D(128, (3,3), strides=(2,2), activation = 'relu', padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.3)(x)

  y = layers.Conv2D(128, (1, 1), strides=(4, 4))(y)
  y = layers.add((x,y))

  # Four more Conv2D+BatchNorm layers without striding
  x = layers.Conv2D(128, (3,3), activation = 'relu', padding="same")(y)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Conv2D(128, (3,3), activation = 'relu', padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.3)(x)

  y = layers.add((x, y))

  x = layers.Conv2D(128, (3,3), activation = 'relu', padding="same")(y)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Conv2D(128, (3,3), activation = 'relu', padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.3)(x)

  y = layers.add((x, y))

  # Max pooling
  pool = layers.MaxPooling2D(pool_size=(4, 4), strides=(4,4))(y)

  # Flatten
  flatten = layers.Flatten()(pool)

  # Dense layers
  dense  = layers.Dense(128, activation="relu")(flatten)
  dense = layers.BatchNormalization()(dense)
  out = layers.Dense(10, activation="softmax")(dense)
  model = Model(inputs=inp, outputs=out)
  return model

def build_model50k():
  model = tf.keras.Sequential([
    layers.Conv2D(16, (5, 5), padding='same', input_shape=(32,32,3)),
    layers.Activation('relu'),

    layers.Conv2D(8, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
  ])
  return model


# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

  val_frac = 0.1
  num_val_samples = int(len(train_images)*val_frac)
    
  val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
  trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
  val_images = train_images[val_idxs, :,:,:]
  train_images = train_images[trn_idxs, :,:,:]

  val_labels = train_labels[val_idxs]
  train_labels = train_labels[trn_idxs]

  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()
  val_labels = val_labels.squeeze()

  input_shape  = train_images.shape[1:]
  train_images = train_images / 255.0
  test_images  = test_images  / 255.0
  val_images   = val_images   / 255.0
  ########################################
  ## Build and train model 1
  model1 = build_model1()
  model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model1.summary()  
  # compile and train model 1.
  # history1 = model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
  

  # model1.save('model1.h5')
  # print("Model 1 saved")
  model1 = tf.keras.models.load_model('model1.h5')  
  print(f"\nModel 1 loaded")
  train_loss, train_acc = model1.evaluate(train_images, train_labels)
  print('Training accuracy:', train_acc)
  val_loss, val_acc = model1.evaluate(val_images, val_labels)
  print('Validation accuracy:', val_acc)
  test_loss, test_acc = model1.evaluate(test_images, test_labels)
  print('Test accuracy:', test_acc)
  image = Image.open(r'C:\Users\hunte\Machine Learning for IoT\simple-convolutional-neural-networks-hwburnt\airplane.jpg')
  image = image.resize((32, 32))  
  image_array = np.array(image) / 255.0
  image_array = np.expand_dims(image_array, axis=0)
  predictions = model1.predict(image_array)
  predicted_class = np.argmax(predictions)
  class_name = class_names[predicted_class]
  image.save(f"model1_test_image_{class_name}.jpg")
  print(f"The predicted class for the image is: {class_name}")

  # Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model2.summary()
  
  # history2 = model2.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
  
  # model2.save('model2.h5')
  # print("Model 2 saved")
  model2 = tf.keras.models.load_model('model2.h5')
  print(f"\nModel 2 loaded")
  train_loss, train_acc = model2.evaluate(train_images, train_labels)
  print('Training accuracy:', train_acc)
  val_loss, val_acc = model2.evaluate(val_images, val_labels)
  print('Validation accuracy:', val_acc)
  test_loss, test_acc = model2.evaluate(test_images, test_labels)
  print('Test accuracy:', test_acc)
  image = Image.open(r'C:\Users\hunte\Machine Learning for IoT\simple-convolutional-neural-networks-hwburnt\airplane.jpg')
  image = image.resize((32, 32))  
  image_array = np.array(image) / 255.0
  image_array = np.expand_dims(image_array, axis=0)
  predictions = model2.predict(image_array)
  predicted_class = np.argmax(predictions)
  class_name = class_names[predicted_class]
  image.save(f"model2_test_image_{class_name}.jpg")
  print(f"The predicted class for the image is: {class_name}")

  model3 = build_model3()
  model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model3.summary()

  # history3 = model3.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=50)
  
  
  # model3.save('model3.h5')
  # print("Model 3 saved")
  model3 = tf.keras.models.load_model('model3.h5')  
  print(f"\nModel 3 loaded")

  train_loss, train_acc = model3.evaluate(test_images,  test_labels, verbose=2)
  print('Training accuracy:', train_acc)
  val_loss, val_acc = model3.evaluate(val_images, val_labels, verbose=2)
  print('Validation accuracy:', val_acc)
  test_loss, test_acc = model3.evaluate(test_images, test_labels, verbose=2)
  print('Test accuracy:', test_acc)

  model50k = build_model50k()
  model50k.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model50k.summary()

  # history50k = model50k.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=50)
  model50k = tf.keras.models.load_model('best_model.h5')

  train_loss, train_acc = model50k.evaluate(test_images,  test_labels, verbose=2)
  print('Training accuracy:', train_acc)
  val_loss, val_acc = model50k.evaluate(val_images, val_labels, verbose=2)
  print('Validation accuracy:', val_acc)
  test_loss, test_acc = model50k.evaluate(test_images, test_labels, verbose=2)
  print('Test accuracy:', test_acc)

  # model50k.save('best_model.h5')
  # print("Best Model saved")
  ### Repeat for model 3 and your best sub-50k params model
  
  
