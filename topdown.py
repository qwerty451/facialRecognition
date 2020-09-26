import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
from cv2 import *
import sys
from tensorflow import keras
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential, save_model, load_model
from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

"""
python filename.py (groupimage/single photo) (face)

Try to make sure that the faces themselves in the images are atleast 224x224

This script uses mctnn and vggface, both are MIT licensed.
https://github.com/ipazc/mtcnn
https://github.com/rcmalli/keras-vggface
"""


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

firstImage = sys.argv[1]
secondImage = sys.argv[2]

#Input is face pixel data, return facial features
def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    model = VGGFace(model='resnet50',
      include_top=False,
      input_shape=(224, 224, 3),
      pooling='avg')

    # perform prediction
    return model.predict(samples)

# Input is the filename, return is the coordinates of al the faces in the photo
def getFaces(filename):
    faces = []
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    
    #Go through every image append and return the processed data
    for result in results:
        x1, y1, width, height = result['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        x1, y1, width, height = result['box']
        face = pixels[y1:y2, x1:x2]
        point_array.append(result['box'])
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = asarray(image)
        faces.append(face_array)
    return faces

point_array = []
test = getFaces(FirstImage)
face = getFaces(secondImage)
score_face = get_model_scores(face)

#Compare the returned features with each face and determine whether or not it's the same face.
counter = 0
for singleFace in test:
    test = np.expand_dims(singleFace, axis=0)
    score_test = get_model_scores(test)
    if cosine(score_test, score_face) <= 0.4:
        print("Faces Matched " + str(counter))
        start_point = point_array[counter]
        print (start_point)

        #Create a red box around the face and save the image.
        x1, y1, width, height = start_point
        x1, y1 = abs(x1), abs(y1)
        im = np.array(Image.open(firstImage), dtype=np.uint8)
        fig,ax = plt.subplots(1)
        ax.imshow(im)
        rect = patches.Rectangle((x1,y1),width,height,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.savefig('output.png', dpi=300, bbox_inches='tight')

        counter += 1
    else:
        print("Faces don't match " + str(counter))
        counter += 1