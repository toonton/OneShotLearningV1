from distutils import command
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from DataOrg import *


#gui and camera setup
root = Tk()
root.geometry("1200x700")
label = Label(root)
label.frame_num = 0
label.grid(row=0, column=0)
cap= cv2.VideoCapture(0)


if not cap.isOpened():
    print("Cannot open camera")
    exit()

def show_frames():
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   label.after(20, show_frames)
    
def key_pressed():
   take_pic()
   

def take_pic():
    global textbox
    userinput = textbox.get()
    if not os.path.exists("C:\\Users\\matan\\OneDrive\\Desktop\\machine learning\\PicturesForDS\\Test_faces\\" + userinput):
        os.makedirs("C:\\Users\\matan\\OneDrive\\Desktop\\machine learning\\PicturesForDS\\Test_faces\\" + userinput)
    if not os.path.exists("C:\\Users\\matan\\OneDrive\\Desktop\\machine learning\\PicturesForDS\\Train_faces\\" + userinput):
        os.makedirs("C:\\Users\\matan\\OneDrive\\Desktop\\machine learning\\PicturesForDS\\Train_faces\\" + userinput)
    file_name = f"{label.frame_num}.png"
    imagetk = label.imgtk
    imgpil = ImageTk.getimage( imagetk )
    imgpil.save("C:\\Users\\matan\\OneDrive\\Desktop\\machine learning\\PicturesForDS\\Train_faces\\" + userinput + "\\" + "0.png", "PNG")
    imgpil.save("C:\\Users\\matan\\OneDrive\\Desktop\\machine learning\\PicturesForDS\\Test_faces\\" + userinput + "\\" + "0.png", "PNG")
    labelpicpic = Label(root)
    labelpicpic.grid(row=0, column=1)
    img = PhotoImage(file=r"C:\\Users\\matan\\OneDrive\\Desktop\\machine learning\\PicturesForDS\\Test_faces\\" + userinput + "\\" + "0.png")
    labelpicpic.configure(image=img)
    labelpicpic.image = img
    label.frame_num += 1
    imgpil.close()
    trainX, testX, trainY, testY = save_comp()
    #data = load('faceme.npz')
    model = load_modeling()
    save_embeding(trainX, testX, trainY, testY, model)
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
    model.fit(trainX, trainy)
    #pickle.dump(model, 'modeldel.joblib')
    print("Added you to the list")

   
def predict_face():
    #model = pickle.load('modeldel.joblib')
    out_encoder = LabelEncoder()
    file_name = f"{label.frame_num}.png"
    imagetk = label.imgtk
    imgpil = ImageTk.getimage( imagetk )
    samples = expand_dims(imgpil, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

#gui objects
labelname = Label(root, text="Add new someone: ^^^")
LastPiclabel = Label(root,text="The last picture that was taken is: ^^^^ ")
YOUU = Label(root,text="Press P to take picture: ^^^^")
textbox=Entry(root, width= 40)
button = Button(root, text="Okay", command=key_pressed)
button2 = Button(root, text="Predict YOU", command=predict_face)

#grids placements
YOUU.grid(row=1,column=0)
LastPiclabel.grid(row=1, column=1)
textbox.grid(row=2, column=0)
labelname.grid(row=3, column=0)
button.grid(row=4, column=0)
button2.grid(row=4, column=1)

show_frames()   
root.mainloop()


#To make a new folder in computer (For new people to class)
#def create():
#   folder = E1.get()
#   newpath = r"C:\Users\....\folder" 
#   if not os.path.exists(newpath):
#       os.makedirs(newpath)