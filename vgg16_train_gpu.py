from tensorflow.keras.applications import VGG16
#from keras.datasets import mnist
#from keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras.layers import Dense,Flatten,Dropout
import cv2
import numpy as np
import tensorflow.compat.v1 as tf_v1
import tensorflow as tf
from tensorflow.keras import layers

data_dir = '/export/home/iceicehyhy/dataset/MNIST_224X224_3/train'
img_height = 224
img_width = 224
batch_size = 32

test_data_dir = '/export/home/iceicehyhy/dataset/MNIST_224X224_3/test'

tf.debugging.set_log_device_placement(True)

try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:2'):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      labels = 'inferred',
      label_mode = 'int',
      color_mode = 'rgb',
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      labels = 'inferred',
      label_mode = 'int',
      color_mode = 'rgb',
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
      test_data_dir,
      labels = 'inferred',
      label_mode = 'int',
      color_mode = 'rgb',
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    #image_batch, labels_batch = next(iter(normalized_ds))

    # You could either manually tune this value, or set it to tf.data.AUTOTUNE, which will prompt the tf.data runtime to tune the value dynamically at runtime.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = normalized_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = normalized_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # manual retrieval
    # for image_batch, labels_batch in train_ds:
    #   print(image_batch.shape)
    #   print(labels_batch.shape)
    #   break

    #建立模型
    print ("creating the model")
    # conv_base=VGG16(weights=None,
    # 				include_top=False,
    # 				input_shape=(224,224,3),
    # 				)

    conv_base = VGG16(weights=None, include_top=False, input_shape=(224,224,3),)
    print ("base model created")
    conv_base.trainable=True
    model= tf.keras.Sequential()
    model.add(conv_base)
    # flatten the input [1, 10, 64]  ---> [640]
    model.add(Flatten())
    model.add(Dense(4096,activation="relu"))
    model.add(Dropout(0.5))
    # layer 14
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    # layer 15
    model.add(Dense(10,activation="softmax"))
    model.summary()

    #编译模型
    print ("compiling the model")
    sgd = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0)
    Nadam = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    adam = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    # using sparse_categorical_crossentropy because the label is in integer form, use categorical_crossentropy if in matrix form
    #model.compile(optimizer=adam,loss=loss,metrics=["accuracy"])
    #print ("model compilied")

    # using sparse_categorical_crossentropy because the label is in integer form, use categorical_crossentropy if in matrix form
    model.compile(optimizer=adam,loss=loss,metrics=["accuracy"])
    print ("model compilied")
    #训练模型
    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=3
    )

    #评估模型
    test_loss,test_acc=model.evaluate(test_ds)
    print("The accuracy is:"+str(test_acc))
except RuntimeError as e:
  print(e)


