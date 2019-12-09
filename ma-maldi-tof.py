print("importing libraries...") # import needed libraries ==========
from collections import OrderedDict
import tensorflow as tf
import csv, os, pandas as pd, re as regex, numpy as np
import xlrd, xlwt, openpyxl
import plotly.offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.core.display import display, HTML
import matplotlib 
from matplotlib import pyplot as plt 
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Activation
from keras.layers import Input, Dense
from keras.utils import np_utils
tf.control_flow_ops = tf # set control flow options to tensorflow - NHWC (must change if backend is theano)
import torch
from torch.autograd import Variable
print("done") # ====================================================




print("gathering filenames...(from google drive)") # Gather relevant filepaths ========
filedir = []
filepaths = []
filepaths_cleaned = []
filepaths_small = []

for dirname, dirnames, filenames in os.walk('./drive/My Drive/Colab Notebooks'):
    # get path to all subdirectories first.
    for subdirname in dirnames:
        #print(os.path.join(dirname, subdirname))        
        if subdirname[-10:] == 'RAWSPECTRA':
          filedir.append(os.path.join(dirname, subdirname))
          #print(os.path.join(dirname, subdirname, filenames[0]))
          #for directory in dirnames:
            #print(directory)
          
for path, folder, fileslist  in os.walk(filedir[0]):
  for file in fileslist:
    if file[-3:] == 'txt' and file[0:3] == 'Dei':
      #print(file)
#     if path[0:17] == "./RAWSPECTRA/P21s" and path[-3:] == 'txt':
      filepaths_small.append(os.path.join(path, file))
      filepaths_cleaned.append(os.path.join(path, file))
      #print(file)
      #print(os.path.join(path, file))

print("")
print("done\n") # ===================================================

print("gathering data...") # Reads/converts data in each file =======
listX = []
listY = []
listXY = []

i = 0
for path in filepaths_small[0:15]: # takes first n files only (filepaths_small[0:5] takes first 5)
    #print(path)
    with open(path) as file:  # creates ambiguous var 
        reader_obj = csv.reader(file, delimiter=' ')
        array_list = list(reader_obj)
        array_list = array_list[:1000] # takes the first 1,000 data sets only (*** NECESSARY TO ENSURE UNIFORM SIZES ***)
    #end with
    listX.append([item[0] for item in array_list]) # buils 2D array of intensities across runs
    listY.append([item[1] for item in array_list]) # bulds 2D array of mass2charge ratios across runs
    listXY.append([item for item in array_list])   # builds 3D array with 2D intensity and mass2charge data across runs
    i += 1
#end for


for i in range(len(listX)):    
    for j in range(len(listX[i])):
        listX[i][j] = float(listX[i][j])
        listY[i][j] = float(listY[i][j])
    #end for    
    listX[i] = np.array(listX[i])
    listY[i] = np.array(listY[i])
    listXY[i] = np.array(listXY[i])
#end for
listX = np.array(listX)
listY = np.array(listY)
listXY = np.array(listXY)

print("")
print(filepaths_cleaned[0:15])
print("...")
print(listXY[0])
print('...')
print("")
print("done\n") # ===================================================


print("building matplotlib plots...") # =============================
%matplotlib inline

m2z_list = []
freq_list = []
spectrum_list = []

for i in range(len(listX)):
    m2z_list.append(listX[i])
    freq_list.append(listY[i])
    spectrum_list.append(listXY[i])
#     plt.figure().suptitle('Time of Flight', fontsize=10)
    plt.figure(figsize=(5, 4)).add_subplot(1,1,1).plot(m2z_list[i], freq_list[i])
    plt.title('Intensity vs mass/Charge' + ': ' + filepaths_small[i][13:-4], fontsize=10)
    plt.xlabel('Mass to charge ratio (m/z)', fontsize=8)
    plt.ylabel('Intensity (I/I_0)', fontsize=8)
    plt.grid(True)
    #fig.savefig('test.jpg')
#end for

print('done') # ====================================================





print("\nbuilding keras model...") # ================================

train_m2z = np.array(listX[0:15]) # choosing number of sets for training 
train_intens = np.array(listY[0:15])
training_spectra = listXY[0:15]
print("input dims = " + str(training_spectra.shape))

data = training_spectra.reshape(len(training_spectra), len(training_spectra[0]), len(training_spectra[0][0]))
img_cols, img_rows = len(data[0]), len(data[0][0])
# labels = np.array([1, 0, 1, 1, 1, 0, 0]) # target/matching array must have same size as training array (here each output is matched with a list of X values)

model = Sequential() # use sequential if model is a simple collection of layers

# filter size = 32, input shape is num of data points in x-dim * data pts in y-dim * num data pts in z-dim
model.add(Convolution2D(32, kernel_size=(2, 2), activation='relu', input_shape=(img_cols, img_rows, 1)))
model.add(Flatten())
model.add(Dense(15)) 
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  

# choose batch size for testing
n_plots = len(data)
  
# set data size (must be total)
# data = np.array(n_plots*2*53288)

# assign labels to plots
#labels = np.random.randint(0,4,n_plots) # chooses a random int from low_val - 0 (inclusive) to high_val - 2 (exclusive) (so 1 or 0)
labels = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
print("labels = " + str(labels))
labels = keras.utils.to_categorical(labels, 15)
# labels = np_utils.to_categorical(list(labels))  # labels must have congruent dimensionality with data
print(len(labels))
print(data.shape, n_plots, len(data[0]), len(data[0][0]))
# add dimension to images (final shape must have same num of elements as initial)
data = data.reshape(n_plots, len(data[0]), len(data[0][0]), 1) 
# =================================================================

# fit data with model
print('\nmodel compiled')
model.fit(data, labels, epochs=10, batch_size=15, verbose=False)  # batch size <= data size,  verbose will print status of each epoch(iteration)
# model.fit(data, labels, epochs=1000, batch_size=10, verbose=False)  # starts training
print('model fit complete')
print("\ncategorized labels are:")
print(labels)
print(len(labels))
# ===================================================================



print('\ntesting input...')
testing_data = data[0]
testing_data = np.expand_dims(testing_data, axis=0)
# testing_data.reshape(len(data[0]), len(data[0][0]), 1)
predictions = model.predict(testing_data, batch_size=10, verbose=1)
print("")
print("prediction is:")
print(predictions)
for p in predictions:
    for index, item in enumerate(p):
        if item > 0:
            print("=> p"+ str(index))
print('done') # ======================================================
