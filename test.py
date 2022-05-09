import os
import cv2 as cv
import numpy as np
from resnet50 import resnet50_model
import pandas as pd

img_width, img_height = 224, 224
num_channels = 3
num_classes = 196
model_weights_path = 'models/model.64-0.87.hdf5'

if __name__ == '__main__':
    model = resnet50_model(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)

    correct = 0
    total_pred = 0
    path = "data/test"
    labels = pd.read_csv("data/test/labels.csv")
    out = open('result.txt', 'a')

    for index, img_class, img_name in labels.itertuples():
        filename = os.path.join(path, img_name)

        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        pred = np.argmax(preds) + 1
        total_pred = total_pred + 1

        if pred == img_class:
            correct = correct + 1
        out.write(f'{pred} {img_class} {prob:.2f}\n')

    print(f'correct predictions:{correct}/{total_pred}')
    print(f'prediction accuracy:{correct / total_pred}')
