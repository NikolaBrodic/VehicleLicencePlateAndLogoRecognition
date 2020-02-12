from darkflow.net.build import TFNet
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage import feature
import numpy as np
import cv2
import os
from datetime import datetime

options = {"pbLoad": "yolo/yolo-plate.pb", "metaLoad": "yolo/yolo-plate.meta", "gpu": 0.7}
yolo_plate_detection = TFNet(options)


def make_hog_data():
    images_dir = 'datasets/HFUT-VL-Logos'

    for folder_name in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, folder_name)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hog = feature.hog(gray, orientations=9, pixels_per_cell=(cell, cell),
                              cells_per_block=(bins, bins), transform_sqrt=True, block_norm="L1")
            data.append(hog)
            labels.append(folder_name)

    start = datetime.now().replace(microsecond=0)
    model.fit(data, labels)
    end = datetime.now().replace(microsecond=0)
    print('SVM score:', round(model.score(data, labels) * 100, 2))
    print('Training duration:', (end - start))


def crop_logo_area(img, predictions):
    predictions = sorted(predictions, key=lambda k: k['confidence'], reverse=True)
    x_top = predictions[0]['topleft']['x']
    y_top = predictions[0]['topleft']['y']
    x_bottom = predictions[0]['bottomright']['x']
    y_bottom = predictions[0]['bottomright']['y']

    w = x_bottom - x_top
    h = y_bottom - y_top
    x_top_logo = int(x_top + w / 2 - h)
    x_bottom_logo = int(x_bottom - w / 2 + h)
    y_top_logo = y_top - w
    y_top_logo = y_top_logo if y_top_logo >= 0 else 0
    y_bottom_logo = y_top + int(h / 6)

    return img[y_top_logo:y_bottom_logo, x_top_logo:x_bottom_logo]


def sliding_window(image, step_size, window_size):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def find_logo():
    images_dir = 'datasets/TestLogos'
    actual_labels = []
    predicted_labels = []
    (win_w, win_h) = (80, 80)

    for folder_name in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, folder_name)
        for img_name in os.listdir(folder_path):
            actual_labels.append(folder_name)
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            yolo_predictions = yolo_plate_detection.return_predict(img)
            logo_area = crop_logo_area(img, yolo_predictions)
            # cv2.imshow('logo area', logo_area)

            bnd_boxes = []
            probabilities = []
            bnd_boxes_65 = []
            probabilities_65 = []
            # first pass with square sliding window
            for (x, y, window) in sliding_window(logo_area, step_size=8, window_size=(win_w, win_h)):
                # if the window does not meet desired size, ignore it
                if window.shape[0] != win_h or window.shape[1] != win_w:
                    continue

                gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                logo = cv2.resize(gray, (64, 64))
                hog = feature.hog(logo, orientations=9, pixels_per_cell=(cell, cell),
                                  cells_per_block=(bins, bins), transform_sqrt=True, block_norm="L1")
                predict_probs = model.predict_proba(hog.reshape(1, -1))
                max_predict_prob = max(predict_probs[0])
                if max_predict_prob >= 0.8:
                    bnd_boxes.append((x, y, x + win_w, y + win_h))
                    probabilities.append(max_predict_prob)

            # second pass with rectangular sliding window
            if not bnd_boxes:
                win_w2 = 110
                for (x, y, window) in sliding_window(logo_area, step_size=5, window_size=(win_w2, win_h)):
                    # if the window does not meet desired size, ignore it
                    if window.shape[0] != win_h or window.shape[1] != win_w2:
                        continue

                    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                    logo = cv2.resize(gray, (64, 64))
                    hog = feature.hog(logo, orientations=9, pixels_per_cell=(cell, cell),
                                      cells_per_block=(bins, bins), transform_sqrt=True, block_norm="L1")
                    predict_probs = model.predict_proba(hog.reshape(1, -1))
                    max_predict_prob = max(predict_probs[0])
                    if max_predict_prob >= 0.8:
                        bnd_boxes.append((x, y, x + win_w2, y + win_h))
                        probabilities.append(max_predict_prob)

            # third pass with rectangular sliding window
            if not bnd_boxes:
                win_w2 = 90
                win_h2 = 60
                for (x, y, window) in sliding_window(logo_area, step_size=5, window_size=(win_w2, win_h2)):
                    # if the window does not meet desired size, ignore it
                    if window.shape[0] != win_h2 or window.shape[1] != win_w2:
                        continue

                    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                    logo = cv2.resize(gray, (64, 64))
                    hog = feature.hog(logo, orientations=9, pixels_per_cell=(cell, cell),
                                      cells_per_block=(bins, bins), transform_sqrt=True, block_norm="L1")
                    predict_probs = model.predict_proba(hog.reshape(1, -1))
                    max_predict_prob = max(predict_probs[0])
                    if max_predict_prob >= 0.8:
                        bnd_boxes.append((x, y, x + win_w2, y + win_h2))
                        probabilities.append(max_predict_prob)
                    if max_predict_prob > 0.65:
                        bnd_boxes_65.append((x, y, x + win_w2, y + win_h2))
                        probabilities_65.append(max_predict_prob)

            if not bnd_boxes:
                bnd_boxes = bnd_boxes_65
                probabilities = probabilities_65

            if not bnd_boxes:
                predicted_labels.append('NONE')
                print(' Actual:   ', actual_labels[-1])
                print(' Predicted:', predicted_labels[-1])
                print('---------------------')
                continue

            max_proba_idx = np.argmax(probabilities)
            (start_x, start_y, end_x, end_y) = bnd_boxes[max_proba_idx]
            logo_img = logo_area[start_y:end_y, start_x:end_x]
            gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
            logo = cv2.resize(gray, (64, 64))
            hog = feature.hog(logo, orientations=9, pixels_per_cell=(cell, cell),
                              cells_per_block=(bins, bins), transform_sqrt=True, block_norm="L1")
            prediction = model.predict(hog.reshape(1, -1))[0].title()
            predicted_labels.append(prediction)
            print(' Actual:   ', actual_labels[-1])
            print(' Predicted:', predicted_labels[-1])
            print('---------------------')

            # cv2.rectangle(logo_area, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            # cv2.imshow("Pick", logo_area)
            #
            # cv2.waitKey(0)

    print('Classification report:')
    print(classification_report(actual_labels, predicted_labels, target_names=os.listdir(images_dir), digits=3))


data = []
labels = []

cell = 8
bins = 2

model = SVC(kernel='linear', probability=True)

make_hog_data()
find_logo()
