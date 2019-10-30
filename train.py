# # Instructions

# There are 3 files with the extension `.npy` in this folder.
# They can be read with `numpy`.

# - `class_a.npy`: Sample images from class A
# - `class_b.npy`: Sample images from class B
# - `field.npy`: Sample images that are similar to the images that will be used for evaluation

# To do

# 1. a training program that generates a model to be used by the classifier program,
# 2. a classifier program that uses the model above to classify images as class A or class B, and
# 3. an evaluation program that evaluates the performance of the classifier program above.

#############################################################################################################

import sys
sys.path.append("NumPy_path")
import numpy as np

######################################################## import dataset and reshape

class_A = np.load('class_a.npy')
class_B = np.load('class_b.npy')
field = np.load('field.npy')

reshapeA = class_A.reshape((1000,40*60))
reshapeB = class_B.reshape((1000,40*60))
fieldClass = field.reshape((200,40*60))
print 'reshape class_A ', reshapeA.shape
print '\nreshape class_B ', reshapeB.shape
print '\nreshape class_field ', fieldClass.shape

class_AB = np.append([reshapeA], [reshapeB], axis=1)
print '\nappend class_AB[0] trainInput --', class_AB[0].shape

####################################################### create labels for classes

dataLen = len(class_A)

label_A = np.empty(dataLen, dtype=object)
label_A[dataLen<dataLen+1]=0 #'a'
# print label_A.shape
label_B = np.empty(dataLen, dtype=object)
label_B[dataLen<dataLen+1]=1 #'b'
# print label_B.shape

label_AB = np.append([label_A], [label_B], axis=1)
label_AB = label_AB.T
print '\nappend label_AB trainOutput --', label_AB.shape

y = label_AB.ravel()
label_AB = np.array(y).astype(int) # str
# print label_AB.tolist()

# import random
# nlabel = np.append([label_A], [label_B], axis=1)
# x = nlabel.ravel()
# nlabel = np.array(x).astype(int) # str
# random.shuffle(nlabel[900:1100])
# print nlabel[900:1100]

####################################################### Gaussian naive_bayes

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Initialize Classifier
G = GaussianNB()

print '\n&&&&&&&&&&&&&& Gaussian naive_bayes Method 1 &&&&&&&&&&&&&&&&'
# Train
G.fit(class_AB[0], label_AB)
# Predict
P1 = G.predict(class_AB[0])
print'\naccuracy of model using original data class_AB--> ', accuracy_score(label_AB, P1)

P2 = G.predict(fieldClass)
print'\naccuracy of trainModel using given "field" Class data--> ', accuracy_score(label_AB[0:200], P2)



print '\n&&&&&&&&&&&&&& Gaussian naive_bayes Method 2 &&&&&&&&&&&&&&&&'
trainData, testData, trainData_labels, testData_labels = train_test_split(class_AB[0],
	label_AB, test_size=0.1, random_state=34)

# Train
G.fit(trainData, trainData_labels)
# Predict
P3 = G.predict(trainData)
print '\naccuracy of model using train_test_split against train_data--> ', accuracy_score(trainData_labels, P3)

P4 = G.predict(fieldClass)
print'\naccuracy of testModel using given "field" Class data--> ', accuracy_score(label_AB[0:200], P4)
P4 = G.predict(testData)
print'\naccuracy of testModel using test data--> ', accuracy_score(testData_labels, P4)

########################################################## KNeighborsClassifier

print '\n&&&&&&&&&&&&&& KNeighborsClassifier Method 3 &&&&&&&&&&&&&&&&'

from sklearn.neighbors import KNeighborsClassifier

K = KNeighborsClassifier(n_neighbors=1)
K.fit(class_AB[0], label_AB)
P5 = K.predict(class_AB[0])
print'\naccuracy of model using original data class_AB--> ', accuracy_score(label_AB, P5)

P6 = K.predict(fieldClass)
print'\naccuracy of KNeighborsClassifier using given "field" Class data--> ', accuracy_score(label_AB[1799:1999], P6)


####################################################### svm classifier 

print '\n&&&&&&&&&&&&&&&&&&&&& LinearSVC Method 4 &&&&&&&&&&&&&&&&&&&&&&&&&&&'

from sklearn.svm import LinearSVC
S = LinearSVC()
S.fit(class_AB[0], label_AB)

P7 = S.predict(class_AB[0])
print'\naccuracy of model using original data class_AB--> ', accuracy_score(label_AB, P7)

P8 = S.predict(fieldClass)
print'\naccuracy of LinearSVC using given "field" Class data--> ', accuracy_score(label_AB[599:799], P8)
