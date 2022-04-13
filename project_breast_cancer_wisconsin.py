# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:46:59 2022

@author: wang
"""


import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import os
import datetime
import seaborn as sns

#read .csv dataset
filepath=r"C:\Users\wang\Desktop\shrdc\DeepLearning\exercise\project\Breast Cancer Wisconsin\data - Copy.csv"
breast_cancer = pd.read_csv(filepath)
breast_cancer.drop(breast_cancer.columns[breast_cancer.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(breast_cancer.head())

#%%
#split features  labels
#drop id 
#numeric label (Benign=0,Malignant=1)

breast_cancer['diagnosis'] = breast_cancer['diagnosis'].replace({'B':0,'M':1})

bc_features = breast_cancer.copy()
bc_features = bc_features.drop(['id'],axis=1)
bc_label = bc_features.pop('diagnosis')
print(bc_features.head())
print(bc_label.head())

#%%
#Split the features and labels into train-validation-test sets
#Using 60:20:20 split

from sklearn.model_selection import train_test_split

SEED = 12345

x_train, x_iter, y_train, y_iter = train_test_split(bc_features, bc_label, test_size=0.4,random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter,y_iter,test_size=0.5,random_state=SEED)


#%%
#standardize data
from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()
standardizer.fit(x_train)
x_train_std = standardizer.transform(x_train)
x_val_std = standardizer.transform(x_val)
x_test_std = standardizer.transform(x_test)

#%%
#model creation
nClass = len(np.unique(y_test))
number_input = x_train_std.shape[-1]
number_output = y_train.shape[-1]
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=number_input))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(16,activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
#model.add(tf.keras.layers.Dense(nClass,activation="softmax"))
model.add(tf.keras.layers.Dense(number_output,activation="softmax"))
model.summary()
tf.keras.utils.plot_model(model, to_file='model_plot.png', 
                          show_shapes=True, show_layer_names=True)

#%%
#model compile
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Define callback functions: EarlyStopping and Tensorboard
base_log_path = r"C:\Users\wang\Desktop\shrdc\DeepLearning\exercise\Tensorboard"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
EPOCHS = 100
BATCH_SIZE=16
history = model.fit(x_train_std,y_train,
                    validation_data = (x_val_std,y_val),
                    batch_size=BATCH_SIZE,epochs=EPOCHS,
                    callbacks=[tb_callback,es_callback])

#%%
#plot training vs validation loss/accuracy

import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = history.epoch

plt.plot(epochs,training_loss,label = 'Training Loss')
plt.plot(epochs,val_loss,label = 'Validation Loss')
plt.title('Training vs validation Loss')
plt.legend()
plt.figure()

plt.plot(epochs,training_accuracy,label = 'Training accuracy')
plt.plot(epochs,val_accuracy,label = 'Validation accuracy')
plt.title('Training vs validation accuracy')
plt.legend()
plt.figure()

#%%
#Evaluate with test data for wild testing
test_result = model.evaluate(x_test_std,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test accuracy = {test_result[1]}")

#%%
# prediction with test data
y_pred = np.argmax(model.predict(x_test_std),axis=-1)
print(y_pred)

#%%

# Making the Confusion Matrix:
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot = True)




