import tensorflow as tf
import matplotlib.pyplot as plt

data_dir = '/home/iceicehyhy/Dataset/MNIST_224X224_3/train'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  labels = 'inferred',
  label_mode='int',
  color_mode='rgb',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(224, 224),
  batch_size=32)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")

plt.show()