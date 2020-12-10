Tugas-11
In [30]: from google.colab import drive drive.mount('/content/drive') Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True). In [31]: cd /content/drive/MyDrive/Colab Notebooks /content/drive/MyDrive/Colab Notebooks In [32]: ls 3scene@ datadaun/ Test1.jpg best_weight_scene_mobileNet.h5 dataset@ Tugas10_Python.ipynb Colab_Notebooks/ model_pemandangan/ In [33]:

import the necessary packages
from keras.models import Sequential from keras.layers.core import Activation from keras.layers.core import Flatten from keras.layers.core import Dense from keras.optimizers import Adam from sklearn.preprocessing import LabelBinarizer from sklearn.model_selection import train_test_split from sklearn.metrics import classification_report from PIL import Image from imutils import paths import numpy as np import os In [34]:

grab all image paths in the input dataset directory, then initialize
our list of images and corresponding class labels
print("[INFO] loading images...") imagePaths = paths.list_images("datadaun") data = [] labels = [] [INFO] loading images... In [35]:

loop over our input images
for imagePath in imagePaths: # load the input image from disk, resize it to 64x64 pixels, scale # the pixel intensities to the range [0, 1], and then update our # images list image = Image.open(imagePath) image = np.array(image.resize((64, 64))) / 255.0 #normalisasi data.append(image)

# extract the class label from the file path and update the
# labels list
label = imagePath.split(os.path.sep)[-2]
labels.append(label)
In [36]: print(labels) ['GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_01', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_02', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_03', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_04', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05', 'GMB_05'] In [37]:

encode the labels, converting them from strings to integers
lb = LabelBinarizer() labels = lb.fit_transform(labels) In [38]: print(labels) [[1 0 0 0 0] [1 0 0 0 0] [1 0 0 0 0] ... [0 0 0 0 1] [0 0 0 0 1] [0 0 0 0 1]] In [39]:

perform a training and testing split, using 75% of the data for
training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25, shuffle=True) print(trainX.shape) print(testX.shape) (438, 64, 64, 3) (146, 64, 64, 3) In [45]: from keras.layers import Convolution2D, MaxPooling2D from keras.models import Sequential from keras.layers.core import Flatten, Dense

model1 = Sequential() model1.add(Convolution2D(8, (3,3), activation='relu', input_shape=(64,64,3))) model1.add(MaxPooling2D(2,2)) model1.add(Convolution2D(16, (3,3), activation='relu')) model1.add(MaxPooling2D(2,2)) #fully connected layer model1.add(Flatten()) model1.add(Dense(100, activation='relu')) model1.add(Dense(5, activation='softmax'))

model1.summary() Model: "sequential_8"

Layer (type) Output Shape Param #
conv2d_12 (Conv2D) (None, 62, 62, 8) 224

max_pooling2d_12 (MaxPooling (None, 31, 31, 8) 0

conv2d_13 (Conv2D) (None, 29, 29, 16) 1168

max_pooling2d_13 (MaxPooling (None, 14, 14, 16) 0

flatten_8 (Flatten) (None, 3136) 0

dense_12 (Dense) (None, 100) 313700

dense_13 (Dense) (None, 5) 505
Total params: 315,597 Trainable params: 315,597 Non-trainable params: 0

In [46]: from keras.models import Sequential from keras.layers.core import Flatten from keras.layers.core import Dense

model = Sequential( [ Flatten(input_shape=(64643,)), Dense(100, activation="relu", name="layer1"), Dense(16, activation="relu", name="layer2"), Dense(16, activation="relu", name="layer3"), Dense(3, activation = "softmax", name="layer4"), ] ) In [47]: model1.summary() Model: "sequential_8"

Layer (type) Output Shape Param #
conv2d_12 (Conv2D) (None, 62, 62, 8) 224

max_pooling2d_12 (MaxPooling (None, 31, 31, 8) 0

conv2d_13 (Conv2D) (None, 29, 29, 16) 1168

max_pooling2d_13 (MaxPooling (None, 14, 14, 16) 0

flatten_8 (Flatten) (None, 3136) 0

dense_12 (Dense) (None, 100) 313700

dense_13 (Dense) (None, 5) 505
Total params: 315,597 Trainable params: 315,597 Non-trainable params: 0

In [60]:

train the model using the Adam optimizer
print("[INFO] training network...") opt = Adam(lr=1e-3, decay=1e-3 / 50) model1.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) H = model1.fit(trainX, trainY, validation_data=(testX, testY), epochs=25, batch_size=32) [INFO] training network... Epoch 1/25 14/14 [==============================] - 1s 76ms/step - loss: 0.0618 - accuracy: 0.9749 - val_loss: 1.2698 - val_accuracy: 0.7329 Epoch 2/25 14/14 [==============================] - 1s 68ms/step - loss: 0.0357 - accuracy: 0.9932 - val_loss: 1.2251 - val_accuracy: 0.7603 Epoch 3/25 14/14 [==============================] - 1s 66ms/step - loss: 0.0142 - accuracy: 0.9954 - val_loss: 1.1844 - val_accuracy: 0.8014 Epoch 4/25 14/14 [==============================] - 1s 66ms/step - loss: 0.0076 - accuracy: 1.0000 - val_loss: 1.1433 - val_accuracy: 0.7877 Epoch 5/25 14/14 [==============================] - 1s 94ms/step - loss: 0.0045 - accuracy: 1.0000 - val_loss: 1.1947 - val_accuracy: 0.7808 Epoch 6/25 14/14 [==============================] - 1s 78ms/step - loss: 0.0039 - accuracy: 1.0000 - val_loss: 1.1397 - val_accuracy: 0.7945 Epoch 7/25 14/14 [==============================] - 1s 67ms/step - loss: 0.0031 - accuracy: 1.0000 - val_loss: 1.1577 - val_accuracy: 0.8014 Epoch 8/25 14/14 [==============================] - 1s 66ms/step - loss: 0.0029 - accuracy: 1.0000 - val_loss: 1.1670 - val_accuracy: 0.8082 Epoch 9/25 14/14 [==============================] - 1s 67ms/step - loss: 0.0026 - accuracy: 1.0000 - val_loss: 1.1739 - val_accuracy: 0.7945 Epoch 10/25 14/14 [==============================] - 1s 68ms/step - loss: 0.0024 - accuracy: 1.0000 - val_loss: 1.1771 - val_accuracy: 0.8014 Epoch 11/25 14/14 [==============================] - 1s 71ms/step - loss: 0.0024 - accuracy: 1.0000 - val_loss: 1.1945 - val_accuracy: 0.8151 Epoch 12/25 14/14 [==============================] - 1s 72ms/step - loss: 0.0022 - accuracy: 1.0000 - val_loss: 1.1927 - val_accuracy: 0.7877 Epoch 13/25 14/14 [==============================] - 1s 76ms/step - loss: 0.0021 - accuracy: 1.0000 - val_loss: 1.2074 - val_accuracy: 0.8082 Epoch 14/25 14/14 [==============================] - 1s 76ms/step - loss: 0.0019 - accuracy: 1.0000 - val_loss: 1.1905 - val_accuracy: 0.7877 Epoch 15/25 14/14 [==============================] - 1s 71ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 1.2214 - val_accuracy: 0.7945 Epoch 16/25 14/14 [==============================] - 1s 68ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 1.2392 - val_accuracy: 0.8288 Epoch 17/25 14/14 [==============================] - 1s 67ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 1.2257 - val_accuracy: 0.7808 Epoch 18/25 14/14 [==============================] - 1s 67ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 1.2337 - val_accuracy: 0.8014 Epoch 19/25 14/14 [==============================] - 1s 67ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 1.2443 - val_accuracy: 0.7945 Epoch 20/25 14/14 [==============================] - 1s 66ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 1.2332 - val_accuracy: 0.7945 Epoch 21/25 14/14 [==============================] - 1s 67ms/step - loss: 0.0013 - accuracy: 1.0000 - val_loss: 1.2464 - val_accuracy: 0.7945 Epoch 22/25 14/14 [==============================] - 1s 67ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 1.2509 - val_accuracy: 0.7945 Epoch 23/25 14/14 [==============================] - 1s 67ms/step - loss: 0.0013 - accuracy: 1.0000 - val_loss: 1.2724 - val_accuracy: 0.8219 Epoch 24/25 14/14 [==============================] - 1s 66ms/step - loss: 0.0013 - accuracy: 1.0000 - val_loss: 1.2652 - val_accuracy: 0.7877 Epoch 25/25 14/14 [==============================] - 1s 67ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 1.2821 - val_accuracy: 0.8151 In [62]: import matplotlib.pyplot as plt

print(H.history.keys())

summarize history for accuracy
plt.plot(H.history['accuracy']) plt.plot(H.history['val_accuracy']) plt.title('model1 accuracy') plt.ylabel('accuracy') plt.xlabel('epoch') plt.legend(['train', 'test'], loc='upper left') plt.show()

summarize history for loss
plt.plot(H.history['loss']) plt.plot(H.history['val_loss']) plt.title('model1 loss') plt.ylabel('loss') plt.xlabel('epoch') plt.legend(['train', 'test'], loc='upper left') plt.show() dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

In [61]: print("[INFO]") predictions = model1.predict(testX, batch_size=32) x=list(map(str,lb.classes_)) print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=x)) [INFO] precision recall f1-score support

  GMB_01       0.76      0.76      0.76        29
  GMB_02       0.71      0.81      0.76        37
  GMB_03       0.82      0.64      0.72        28
  GMB_04       0.88      0.92      0.90        25
  GMB_05       0.96      0.96      0.96        27

accuracy                           0.82       146
macro avg 0.83 0.82 0.82 146 weighted avg 0.82 0.82 0.81 146

In [63]: model1.save('model_daun') WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version. Instructions for updating: This property should not be used in TensorFlow 2.0, as updates are applied automatically. INFO:tensorflow:Assets written to: model_daun/assets In [66]: import cv2 import matplotlib.pyplot as plt image1='testdaun.jpg' In [67]: img_array = cv2.imread(image1) plt.imshow(img_array) plt.show()

print(type(img_array))
In [69]: image_testing = Image.open('testdaun.jpg') image_testing = np.array(image_testing.resize((64, 64))) / 255.0 image_testing.shape Out[69]: (64, 64, 3) In [70]: image_testing = np.expand_dims(image_testing, axis=0) print(image_testing.shape) (1, 64, 64, 3) In [71]: output = model1.predict(image_testing, 1) print(output) print(lb.classes_[output.argmax(axis=1)]) [[6.3822817e-08 4.0504753e-07 1.3266290e-15 9.9999118e-01 8.3379609e-06]] ['GMB_04']
