from pyexpat import model
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from numpy import savez_compressed
from mtcnn import MTCNN
from os import listdir
from os.path import isdir
from PIL import Image
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
from DataOrg import *

# develop a classifier for the 5 Celebrity Faces Dataset
# load faces
trainX, testX, trainY, testY = save_comp()
data = load('faceme.npz')
testX_faces = data['arr_2']
model = load_modeling()
save_embeding(trainX, testX, trainY, testY, model)
# load face embeddings
data = load('faceme-embed.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
#test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
#pyplot.imshow(pixels)
#title = '%s (%.3f)' % (predict_names[0], class_probability)
#pyplot.title(title)
#pyplot.show()