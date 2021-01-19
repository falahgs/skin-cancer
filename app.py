import os
import re
import numpy as np
from fastai.vision.all import *
#import aiohttp
#import asyncio
#path = Path(__file__).parent
#import tensorflow as tf
#import keras
#from tensorflow import keras
#from tensorflow.keras import backend as K
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import load_img
#from keras.models import load_model
#import cv2
import uuid
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.
multiple=[]
result=[]
UPLOAD_FOLDER = 'uploads'
# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Model saved with Keras model.save()
#MODEL_PATH = './models/artist2020.h5'

#download from dropbox 
#export_file_url = 'https://www.dropbox.com/s/jon8u6mt8wbzt0f/export.pkl?dl=0'
#export_file_name = 'export.pkl'
#folder_path='./model/'
#import requests
#headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
#r = requests.get(export_file_url, stream=True, headers=headers)
#with open(export_file_name, 'wb') as f:
   # for chunk in r.iter_content(chunk_size=1024): 
       # if chunk:
          #  f.write(chunk)
#
model_mask='./model/export_skin_cancer.pkl'
learn_inf = load_learner(model_mask)
print('Model loading...')
#model=load_model(MODEL_PATH)
print('Model loaded. Started serving...')
def model_predict(img_path,learn_inf):
	#image = cv2.imread(img_path)
	#image = cv2.resize(image, (224, 224))
	#image = image.astype("float") / 255.0
	#image = img_to_array(image)
	#image = np.expand_dims(image, axis=0)
	
	#proba = learn_inf.predict(image)
	results=learn_inf.predict(img_path)
	class_name=results[0]
	c = results[2].numpy()
	print(c)
	np.argmax(c)
	print(np.argmax(c))
	print('accurecy:',c[np.argmax(c)])
	confidence = c[np.argmax(c)] * 100
	return class_name,confidence
@app.route('/', methods=['GET'])
def index():
	# Main page
	for i in glob.glob("uploads/*.jpg"):
		os.remove(i)
	for i in glob.glob("uploads/*.jpeg"):
		os.remove(i)
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	multiple=[]
	if request.method == 'POST':
			# Get the file from post request
		f = request.files['image']
		filename = secure_filename(f.filename)
					# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)
		filename = my_random_string(6) + filename
		os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
		print("file_path:",file_path)
		print('file name:',filename)
		print('Begin Model Prediction...')
		filename='./uploads/'+filename
					# Make prediction
		class_name,confidence= model_predict(filename,learn_inf)

					# Process your result for human
		#pred_class = preds.argmax(axis=1)[0]
		print('class name:',class_name)
		#preds = preds.argmax(axis=1)[0]
		#labels=("Francisco Goya", "Pablo Picasso","Vincent van Gogh")
		#print('preds;',preds)
		print('confidence:',confidence)
		#print('the class:',labels[preds])
		#result = labels[preds]
		#print('result:',result)
		data=',Prediction Confidence:'.join([str(class_name),str(confidence)+'%'])
		print('The class name Info.:',data)
		result=data
		# Convert to string
		os.remove(filename)
		return result
	return None
from werkzeug.serving import run_simple
if __name__ == "__main__":
    run_simple(
    '127.0.0.1',
    9000, app, use_reloader=False, use_debugger=True,
    use_evalex=True, extra_files=None, reloader_interval=1,
    reloader_type='auto', threaded=False, 
    processes=1, request_handler=None, 
    static_files=None, passthrough_errors=False, 
    ssl_context=None)      