import json
import os
import random

import cv2
import numpy as np
import scipy.io
import keras.backend

from resnet50 import resnet50_model

if __name__ == '__main__':

    model_weights_path = 'models/model.64-0.87.hdf5'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196

    model = resnet50_model(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    test_path = 'data/test/'
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]

    num_samples = 20
    samples = random.sample(test_images, num_samples)
    results = []

    for i, image_name in enumerate(samples):
        filename = os.path.join(test_path, image_name)

        img = cv2.imread(filename)
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_CUBIC)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        prediction = model.predict(rgb_img)
        prob = np.max(prediction)
        class_id = np.argmax(prediction)
        print('Predict: {}; class: {}'.format(class_names[class_id][0][0], class_id))
        results.append({'label': class_names[class_id][0][0], 'probability': '{:.4}'.format(prob), 'picture:': filename})
        cv2.imwrite('./images/{}_out.png'.format(i), img)

    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

    keras.backend.clear_session()
