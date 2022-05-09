import os
import cv2 as cv
import keras.backend as K
import numpy as np
from os import listdir
from os.path import isfile, join
from resnet50 import resnet50_model

img_width, img_height = 224, 224
num_channels = 3
num_classes = 196
model_weights_path = 'models/model.64-0.87.hdf5'


if __name__ == '__main__':
    model = resnet50_model(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)

    correct = 0
    total_pred = 0
    out = open('result_validation.txt', 'a')
    for i in range(num_classes):
        path = os.path.join('data/valid/', '%04d' % (i + 1))

        files = [f for f in listdir(path) if isfile(join(path, f))]
        for filename in files:
                filename = path + '/' + filename
                bgr_img = cv.imread(filename)
                rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
                rgb_img = np.expand_dims(rgb_img, 0)
                preds = model.predict(rgb_img)
                prob = np.max(preds)
                class_id = np.argmax(preds)
                pred = class_id + 1
                total_pred = total_pred + 1
                true_label = i + 1
                if pred == true_label:
                    correct = correct + 1
                
                out.write('{}\n'.format(str(pred) + ' ' + str(true_label)))

    out.close()
    K.clear_session()
    
    print('correct predictions:', str(correct))
    print('prediction acurracy:', str(correct/total_pred))
