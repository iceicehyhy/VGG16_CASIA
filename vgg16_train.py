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

# #加载数据
# print ("loading data")
# (x_train,y_train),(x_test,y_test)=mnist.load_data()
# #VGG16模型,权重由ImageNet训练而来，模型的默认输入尺寸是224x224，但是最小是48x48
# #修改数据集的尺寸、将灰度图像转换为rgb图像

# print ("data pre-processing")
# x_train=[cv2.cvtColor(cv2.resize(i,(224,224)),cv2.COLOR_GRAY2BGR)for i in x_train]
# x_test=[cv2.cvtColor(cv2.resize(i,(224,224)),cv2.COLOR_GRAY2BGR)for i in x_test]
# #第一步：通过np.newaxis函数把每一个图片增加一个维度变成(1,48,48,3)。所以就有了程序中的arr[np.newaxis]。
# #第二步：通过np.concatenate把每个数组连接起来组成一个新的x_train数组，连接后的x_train数组shape为(10000,48,48,3)
# print ("concatenating")
# x_train_expand = tf.expand_dims(x) for 
# # x_train=np.concatenate([arr[np.newaxis]for arr in x_train])
# # x_test=np.concatenate([arr[np.newaxis]for arr in x_test])

# print ("convert to [0,1]")
# x_train=x_train.astype("float32")/255
# x_train=x_train.reshape((60000,224,224,3))

# x_test=x_test.astype("float32")/255
# x_test=x_test.reshape((10000,224,224,3))

# # convert class vector(int) to matrix
# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)

# print ("divide training and validation data")
# #划出验证集, for validation
# x_val=x_train[:10000]
# y_val=y_train[:10000]
# x_train=x_train[10000:]
# y_train=y_train[10000:]

data_dir = '/home/iceicehyhy/Dataset/CASIA_224X224'
img_height = 224
img_width = 224
batch_size = 8

#test_data_dir = '/export/home/iceicehyhy/dataset/MNIST_224X224_3/test'
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

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   test_data_dir,
#   labels = 'inferred',
#   label_mode = 'int',
#   color_mode = 'rgb',
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

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

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3),)
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
model.add(Dense(10575,activation="softmax"))
model.summary()

#编译模型
print ("compiling the model")
sgd = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0)
Nadam = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
adam = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False)
# using sparse_categorical_crossentropy because the label is in integer form, use categorical_crossentropy if in matrix form
model.compile(optimizer=adam,loss=loss,metrics=["accuracy"])
print ("model compilied")
#训练模型
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

# #评估模型
# test_loss,test_acc=model.evaluate(test_ds)
# print("The accuracy is:"+str(test_acc))

