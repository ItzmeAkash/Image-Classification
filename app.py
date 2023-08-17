import numpy as np
import sys
import os
import glob
import re
#Keras

import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from keras.preprocessing import image
from keras.models import load_model
from werkzeug.utils import secure_filename


# Flask utils
from flask import Flask,redirect,url_for,request,render_template
from PIL import Image


# Define a flask app
app=Flask(__name__)
model_path='Vgg19.h5'

##load model
model=load_model(model_path)

# model._make_predict_function()

##Preprocessing functions
def model_predict(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
     
    # Preprocessing the image
    x=image.img_to_array(img)
    
    y=np.expand_dims(x,axis=0)
    
    # be carefull how your trained model deals with the input
    # Otherwise, it won't make correct predictions!
    
    c=preprocess_input(y)
    
    preds=model.predict(c)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])

def upload():
    if request.method=="POST":
        
        ##Get the File from the Post
        f=request.files['file']
        #Save the file to ./uploads
        
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        ###Here we make the prediction
        pred=model_predict(file_path,model)
        
        pred_cls=decode_predictions(pred,top=1) #ImageNet Decode
        result=str(pred_cls[0][0][1])           #Convert to String
        return result
    return None
        
        
if __name__ == '__main__':
    app.run(debug=True)