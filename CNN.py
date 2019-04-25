'''
POKEMON NEURAL NET
Program reads in pictures, preprocesses them and runs them on a mlp or CNN. Also processes pictures through pca if u want
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
from tempfile import TemporaryFile
__DIMENSION = 3
__NUMCLUSTERS = 2
def f(seq): # Order preserving
  ''' Modified version of Dave Kirby solution '''
  seen = set()
  return [x for x in seq if x not in seen and not seen.add(x)]

def normalize(x):
        return (x.astype(float) - 127) / 127
def centeralize(x):
    return (x.astype(float) - np.mean(x)) / np.std(x)
def deFuzz(U):
    labels = np.zeros((U.shape[0],))
    i = 0
    for x  in U:
        labels[i] = np.argmax(x)
        i += 1
    return labels

def seperateImages(image , labels , numClusters ):
    list = [ np.zeros((image.shape[0] , __DIMENSION)) for x in range(0,numClusters) ]
    print(image.shape[0])
    i = 0
    for pixel , label in zip(image , labels):
        img = list[int(label)]
        img[i] = pixel
        i += 1
    return list

#Removes whitespace and PCA's the image
def proc(img, filename):
    info = np.iinfo(img.dtype)
    img = img.astype(np.float) / info.max
    im_re = np.reshape(img , (img.shape[0] * img.shape[1] , __DIMENSION))
    #print(im_re)
    #print(im_re)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.transpose(im_re), __NUMCLUSTERS, 2 ,error=0.00001, maxiter=10000, init=None)
    #print(np.transpose(u))
    labels = deFuzz(np.transpose(u))
   # print(labels)
    num = {}
    for l in labels:
        try:
            num[int(l)]+=1
        except:
            num[int(l)] = 1

    cluster = 0
    if num[0] < num[1]:
        cluster = 0
    else:
        cluster = 1

    reduced = np.zeros((num[cluster] , __DIMENSION))

    rIdx = 0
    for i , pixel in enumerate(im_re):
        if labels[i] == cluster:
            reduced[rIdx][0] = pixel[0]
            reduced[rIdx][1] = pixel[1]
            reduced[rIdx][2] = pixel[2]
            rIdx += 1
    rT = np.transpose(reduced)
    #print(rT.shape)
    pca = PCA(n_components=100)
    pca.fit(rT[i])
    pca_img = pca.transform(rT)
    pca_img =  np.transpose(pca_img)
    #outfile = TemporaryFile()
    print(filename)
    np.save('proc/' + str(filename), pca_img)
    return pca_img

#/============================================================/#
if __name__ == "__main__":

    model = Sequential()
    #inports images from same folder
    path = 'POKEMON/'
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
        if '0' in i:
            im = im + 1
#            print(i)
            imgList.append(cv2.imread(i))
            imgList[im-1] = cv2.cvtColor(imgList[im-1], cv2.COLOR_BGR2LAB)
            #imgList[im-1] = cv2.resize(imgList[im-1], (107, 107))
#    imgList[imgList == 255] = 0
    #reads in types csv's
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
    #data augmentation
    for i, value in enumerate(thing):
        for j, value2 in enumerate(un):
            if thing[i] == un[j]:
                imgList.append(cv2.flip(imgList[i], 0))
                thing.append(int(j))
                imgList.append(cv2.flip(imgList[i], 1))
                thing.append(int(j))
                temp = cv2.flip(imgList[i], 1)
                imgList.append(cv2.flip(temp, 0))
                thing.append(int(j))
                thing[i] = int(j)
    x = np.array(imgList)
    y = np.array(thing)
    np.save('labels', y)
    #uncomment to run pca and fuzzy clustering
#    for i in range(0,6407):
#        x[i] = proc(x[i], i)
    
    input_shape = x[1].shape
    num_classes = 18
#    x = centeralize(x)
    x = normalize(x)
    x, x_test, y, y_test = train_test_split(x, y, test_size=.25)
#    print(y)
    unTr = np.unique(y)
    unTe = np.unique(y_test)
    #uncomment when running pca
    #dimData = np.prod(x.shape[1:])
    #x = x.reshape(x, dimData)
    #x_test = x_test.reshape(x_test, dimData)
    
    #comment out if running mlp
# 64 channels, filter window is 5x5, strides are 1x1.
# Input shape is the size of the image.
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
 
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    #model.add(Dense(80, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    '''
    model.add(Dense(512, activation='relu', input_shape=dimData))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    '''
    model.compile(keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),loss='categorical_crossentropy',metrics=['accuracy'])

    labels = to_categorical(y, len(unTr))
    model.fit(x, labels, epochs=10, batch_size=64)
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