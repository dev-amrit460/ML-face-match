import streamlit as sl
from PIL import Image
import os
import cv2
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from mtcnn import MTCNN

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = pickle.load(open('embedding.pkl', 'rb'))
nameList = pickle.load(open('nameList.pkl', 'rb'))
detector = MTCNN()


def upload_img(data):
    try:
        with open(os.path.join('uploads', data.name), 'wb') as f:
            f.write(data.getbuffer())
        return True
    except:
        return False


def feature_extract(img_path, model, detector):
    test_img = cv2.imread(img_path)
    results = detector.detect_faces(test_img)
    x, y, width, height = results[0]['box']

    face = test_img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)

    out = model.predict(preprocess_input(expanded_img)).flatten()
    return out


def match(feature):
    sim, inx = 0, 0
    i: int
    for i in range(len(feature_list)):
        if sim < cosine_similarity(feature.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0]:
            sim = cosine_similarity(feature.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0]
            inx = i
    return [sim, inx]


sl.title('Bollywood Celebrity AI Match')

data = sl.file_uploader('Pick An Image')

if data is not None:
    if upload_img(data):
        display = Image.open(data)

        feature = feature_extract(os.path.join('uploads', data.name), model, detector)
        temp = match(feature)
        ind = temp[1]
        sim = temp[0]

        col1, col2 = sl.columns(2)
        with col1:
            sim = int(sim*100)
            k = 'YOU' + ' ' + str(sim) + ' ' + '%' + ' ' + 'LOOK LIKE'
            sl.caption(k)
            sl.image(display, width=250)
        with col2:
            p = " ".join(nameList[ind].split('\\')[1].split('_')).upper()
            sl.caption(p)
            test = "".join(nameList[ind].split('\\')[1])
            test = os.path.normpath(os.path.join('data', test, "img.jpg"))
            sl.image(test, width=250)
