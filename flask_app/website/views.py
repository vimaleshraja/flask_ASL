from flask import Blueprint, render_template, request
from tensorflow import keras
import numpy as np
from PIL import Image


views = Blueprint('views', __name__)
model = keras.models.load_model('website/cnn_5.h5')


@views.route('/', methods = ['GET'])
def home():
    return render_template('base.html')

@views.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        image = request.files['image']
        image = Image.open(image.stream)
        image = image.resize((64,64))
        # image = keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
        image = keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1,) + image.shape)
        prediction = model.predict(image)

        output = np.argmax(prediction[0])
        labels = list('abcdefghijklmnopqrstuvwxyz')
        labels += ["del", "nothing", "space"]

        return render_template('base.html', prediction_text='ASL Sign is $ {}'.format(labels[output]))
    
    return render_template('base.html')