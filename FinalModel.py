import os, sys, os.path,  math, shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from Create_Sets import *
import random



class_names = ['assorted', 'cardboard','cardboard_and_paper', 'metal_cans','plastic_bags','paper','plastic_bottles', 'shells']

def find_images():
   file_locs = np.array(['lt', 'ty','u','bc'])
   for folder in os.listdir(os.getcwd()):
      if(folder[-3] == "." or  folder[-4] == "." or folder == "valid" or folder == "test" or folder == "train"):
         continue
      #print(1)
      for frames in os.listdir(folder):
         #print(2)
         if(frames == 'Create_Sets.cpython-36.pyc' or frames == 'Split_Extraction.cpython-36.pyc'):
            continue
         for image in os.listdir(folder+'/'+frames):
            #print(3)
            #print(os.getcwd() +'/' +folder +'/'+ frames+ '/'+ image)
            string = folder +'/'+frames+'/' + image
            string2 = frames + image
            tuple = (np.array([string, folder, image, string2]))
            file_locs = np.vstack((file_locs, tuple))
            
   return file_locs


def split(file_locs, class_names):
   class_count = np.empty(len(class_names))
   for item in file_locs:
      class_count[class_names.index(item[1])]  +=1
   np.random.shuffle(file_locs)
   training_count  = (class_count*.8).astype(int)
   training_array = np.array(['t','t','t','t'])
   validation_array = np.array(['t','t','t','t'])
   count=0
   for x in file_locs:
      if training_count[class_names.index(x[1])] > 0:
         training_array = np.vstack((training_array,x))
         training_count[class_names.index(x[1])] -= 1
      else:
         validation_array = np.vstack((validation_array,x))
         
   #print(len(training_array))
   #print(len(validation_array))
   return training_array, validation_array

def create_folders(train_array, valid_array, class_names):   
   if os.path.exists('./train'):
      shutil.rmtree('./train')
   if os.path.exists('./test'):
      shutil.rmtree('./test')
   if os.path.exists('./valid'):
      shutil.rmtree('./valid')
   os.mkdir('./train')
   os.mkdir('./valid')
   for i in class_names:
      os.mkdir('./train/' + i)
      os.mkdir('./valid/' + i)

   for x in train_array:
       shutil.copyfile(x[0], './train/'+ x[1]+'/'+ x[3] )
   for x in valid_array:
      shutil.copyfile(x[0], './valid/'+x[1]+'/'+ x[3] )


train_path = './train'
valid_path = './valid'
batch_size = 100
train_steps= 30
val_steps=10
epochs=10

model = models.Sequential()
model.add(layers.Conv2D(32, (2,2), activation='relu', input_shape=(224,224,3), data_format ='channels_first'))

for i in range(2):
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

arr = find_images()
arr = arr[1:]
train,  valid = split(arr, class_names)
train = train[1:]
valid = valid[1:]
create_folders(train, valid, class_names)
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=class_names, batch_size=batch_size)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=class_names, batch_size=batch_size)
history = model.fit((item for item in train_batches), batch_size=batch_size, epochs = epochs,steps_per_epoch=train_steps, verbose=1, validation_data= (item for item in valid_batches), validation_steps=val_steps)
model.save(os.getcwd()+'/'+sys.argv[1]+'.h5')

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy = history_dict['acc']
val_accuracy = history_dict['val_acc']

for_num = 100

#graph_num = epochs*(for_num + 1)

for i in range(for_num):
   arr = find_images()
   arr = arr[1:]
   train,  valid = split(arr, class_names)
   train = train[1:]
   valid = valid[1:]
   create_folders(train, valid, class_names)
   train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=class_names, batch_size=batch_size)
   valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=class_names, batch_size=batch_size)
   history = model.fit((item for item in train_batches), batch_size=batch_size, epochs = epochs,steps_per_epoch=train_steps, verbose=1, validation_data= (item for item in valid_batches), validation_steps=val_steps)
   history_dict = history.history
   #for j in history_dict['loss']:
    #  loss_values = np.append(loss_values, j)
   #for j in history_dict['val_loss']:
    #  val_loss_values = np.append(val_loss_values, j)
   #for j in history_dict['acc']:
    #  accurracy = np.append(accuracy, j)
   #for j in history_dict['val_acc']:
    #  val_accuracy = np.append(val_accuracy, j)
   model.save(os.getcwd()+'/'+sys.argv[1]+str(i)+'.h5')
   

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy = history_dict['acc']
val_accuracy = history_dict['val_acc']
   
epochs = range(1, len(loss_values)+1)
fig, ax = plt.subplots(1,2,figsize=(14,6))

ax[0].plot(epochs, accuracy, 'bo', label ='Training accuracy')
ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
ax[0].set_title('Training &amp; Validation Accuracy', fontsize = 14)
ax[0].set_xlabel('Epochs',fontsize=16)
ax[0].set_ylabel('Accuracy',fontsize=16)
ax[0].legend()
          
ax[1].plot(epochs,loss_values, 'bo', label ='Training loss')
ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
ax[1].set_title('Training &amp; Validation Loss', fontsize = 14)
ax[1].set_xlabel('Epochs',fontsize=16)
ax[1].set_ylabel('Loss',fontsize=16)
ax[1].legend()

plt.show()
