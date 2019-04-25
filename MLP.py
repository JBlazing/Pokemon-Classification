'''
run on pca'd and fuzzy dataset, mlp only
'''
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import *
from keras.utils import *
import numpy as np
from sklearn.decomposition import PCA
from numpy import genfromtxt
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import *
import cv2
import os
import csv
import skfuzzy as fuzz
from sklearn.metrics import *    
def f(seq): # Order preserving
  ''' Modified version of Dave Kirby solution '''
  seen = set()
  return [x for x in seq if x not in seen and not seen.add(x)]
#/============================================================/#
if __name__ == "__main__":
    model = Sequential()
    path = 'proc/'
    files = os.fsencode(path)
    name = []
    for file in os.listdir(files):
        name.append(path + os.fsdecode(file))
    
    #images might not be in order, need to sort them
    name.sort();
    imgList = []
    im = 0
    #reads in images
    for i in name:
        imgList.append(np.load(i))
    x = np.array(imgList)
    y = np.load('labels.npy')
    x, x_test, y, y_test = train_test_split(x, y[:-1], test_size=.25)
#    print(y)
    unTr = np.unique(y)
    unTe = np.unique(y_test)
    dimData = np.prod(x.shape[1:])
    x = x.reshape(x.shape[0], dimData)
    x_test = x_test.reshape(x_test.shape[0], dimData)
    num_classes = 18
    thing = []
    with open('pokemans.csv') as csvfile:
        inp = csv.reader(csvfile, delimiter=',')
        for row in inp:
            if len(row) > 1:
                thing.append(row[1])
            else:
                thing.append(row[0])
    with open('pokemans.csv') as csvfile:
        inp = csv.reader(csvfile, delimiter=',')
        for row in inp:
            if len(row) > 1:
                thing.append(row[1])
            else:
                thing.append(row[0])

    un = f(thing)
    un = list(un)

    model.add(Dense(512, activation='relu', input_shape=(9, )))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),loss='categorical_crossentropy',metrics=['accuracy'])

    labels = to_categorical(y, len(unTr))
    model.fit(x, labels, epochs=50000, batch_size=32)
    model.save('model.h5')
    labels_test = to_categorical(y_test, 18)
    predi = model.predict(x_test, 32, 1, steps=None)
    print(predi)
    maxi = np.argmax(predi, axis=1)
    print(maxi)
    right = 0
    for i, value in enumerate(maxi):    
        if int(y_test[i]) == maxi[i]:
            right = right + 1
    print(right/len(y_test))

    final=[]
    for i in range(0,len(y_test)):
        final.append(int(y_test[i]))

    unF = list(f(final))
    TotalReport = classification_report(maxi,final)

    # function to get overall precision
    overall_prec = precision_score(maxi, final,average='weighted')

    #this function will allow you to get the precision for each individual class
    prec = precision_score(maxi, final,average=None)

    # this is how you would print or access the precision for the specified class
    num = 0
    for i in prec:
        print(un[unF[num]])
        num = num + 1
        print(i)