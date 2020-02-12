from darkflow.net.build import TFNet
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import cv2
import os
import imutils
from statistics import mean
from sklearn.metrics import accuracy_score
from fuzzywuzzy import fuzz

options = {"pbLoad": "yolo/yolo-plate.pb", "metaLoad": "yolo/yolo-plate.meta", "gpu": 0.7}
yolo_plate_detection = TFNet(options)
print('> Loaded YOLO from disk <')

char_rec_model = tf.keras.models.load_model('models/character_recognition_cnn.h5')
print('> Loaded model from disk <')


def crop_plate(img, predicts):
    predicts = sorted(predicts, key=lambda k: k['confidence'], reverse=True)
    x_top = predicts[0]['topleft']['x']
    y_top = predicts[0]['topleft']['y']
    x_bottom = predicts[0]['bottomright']['x']
    y_bottom = predicts[0]['bottomright']['y']

    plate_only = img[y_top:y_bottom, x_top:x_bottom]

    return plate_only


# compare if images are equal
def are_equal(img1, img2):
    if img1.shape != img2.shape:
        return False
    difference = cv2.subtract(img1, img2)
    b, g, r = cv2.split(difference)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True

    return False


def find_new_rect(char):
    gray = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    thresh_inv = cv2.adaptiveThreshold(gray_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)
    edged = imutils.auto_canny(thresh_inv)
    _, ctrs, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in ctrs]
    biggest_contour = max(contour_sizes, key=lambda k: k[0])[1]

    return cv2.boundingRect(biggest_contour)


def read_plate_characters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)  # smoothing without removing edges.
    thresh_inv = cv2.adaptiveThreshold(gray_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)
    edged = imutils.auto_canny(thresh_inv)
    # cv2.imshow('canny', edged)
    _, ctrs, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours by x coordinate of bounding rectangle
    sorted_ctrs = sorted(ctrs, key=lambda k: cv2.boundingRect(k)[0])
    img_h, img_w, _ = img.shape

    # eliminate too big and too small rectangles
    rect_area_list = []
    non_duplicate_ctrs = []
    for i, contour in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(contour)
        if x / img_w > 0.04:
            if 0.034 < (w / img_w) < 0.15:
                if 0.235 < (h / img_h) < 0.62:
                    rect_area = img[y:y + h, x:x + w]
                    if len(rect_area_list) == 0:
                        rect_area_list.append(rect_area)
                        non_duplicate_ctrs.append(contour)
                    elif not are_equal(rect_area, rect_area_list[-1]):
                        rect_area_list.append(rect_area)
                        non_duplicate_ctrs.append(contour)

    # eliminate inner rectangles
    outer_rects = [cv2.boundingRect(non_duplicate_ctrs[0])]
    for i in range(1, len(non_duplicate_ctrs)):
        j = i - 1
        contour1 = non_duplicate_ctrs[j]
        (x1, y1, w1, h1) = cv2.boundingRect(contour1)
        contour2 = non_duplicate_ctrs[i]
        (x2, y2, w2, h2) = cv2.boundingRect(contour2)
        if not (x2 > x1 and x2 + w2 < x1 + int(1.1 * w1) and y2 > y1 and y2 + h2 < y1 + h1):
            outer_rects.append((x2, y2, w2, h2))

    # join overlapping rectangles
    i = 0
    final_rects = []
    while i < len(outer_rects) - 1:
        j = i + 1
        (x1, y1, w1, h1) = outer_rects[i]
        (x2, y2, w2, h2) = outer_rects[j]
        if x2 >= x1 and x2 + w2 > x1:
            if x2 + w2 <= x1 + w1:
                # join if one is inside another by x axis
                y = min(y1, y2)
                h = y1 + h1 - y if y1 + h1 > y2 + h2 else y2 + h2 - y
                final_rects.append((x1, y, w1, h))
                i += 1
                if i == len(outer_rects) - 2:
                    final_rects.append(outer_rects[-1])
            else:
                w_ol = x1 + w1 - x2
                if (w_ol / w1 > 0.33) or (w_ol / w2 > 0.33):
                    # join only if overlapping is 1/3 of the width
                    w = x2 + w2 - x1
                    y = min(y1, y2)
                    h = y1 + h1 - y if y1 + h1 > y2 + h2 else y2 + h2 - y
                    final_rects.append((x1, y, w, h))
                    i += 1
                    if i == len(outer_rects) - 2:
                        final_rects.append(outer_rects[-1])
                else:
                    final_rects.append((x1, y1, w1, h1))
                    if i == len(outer_rects) - 2:
                        final_rects.append(outer_rects[-1])
        else:
            final_rects.append((x1, y1, w1, h1))
            if i == len(outer_rects) - 2:
                final_rects.append(outer_rects[-1])
        i += 1

    w_mean = mean(rect[2] for rect in final_rects)
    h_mean = mean(rect[3] for rect in final_rects)

    # draw bounding rectangles
    char_list = []
    for i, rect in enumerate(final_rects):
        x, y, w, h = rect
        char = img[y:y + h, x:x + w]
        if w > w_mean * 1.2 and h > h_mean * 1.2:
            # if rectangle is bigger than average, find smaller that better fits character
            (new_x, new_y, new_w, new_h) = find_new_rect(char)
            char = char[new_y:new_y + new_h, new_x:new_x + new_w]
            # # find new rectangle in the plate image
            # res = cv2.matchTemplate(img, char, cv2.TM_CCOEFF_NORMED)
            # _, _, _, top_left = cv2.minMaxLoc(res)
            # x, y, w, h = top_left[0], top_left[1], new_w, new_h

        char_list.append(character_recognition(char))
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv2.imshow('Character segmentation', img)
    char_string = ''.join(char_list)
    return char_string


def character_recognition(img):
    dictionary = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',
                  11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K',
                  21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V',
                  31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}

    char_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    char_gray = cv2.resize(char_gray, (75, 100))
    resized_img = char_gray.reshape((1, 100, 75, 1))
    resized_img = resized_img / 255.0
    new_predictions = char_rec_model.predict(resized_img)
    char = np.argmax(new_predictions)
    return dictionary[char]


def load_images_and_labels():
    img_dir = 'datasets/BelgianLicencePlates/TestPlates'
    labels_dir = 'datasets/BelgianLicencePlates/TestPlatesLabels'
    imgs = []
    lbls = []

    for xml_name in os.listdir(labels_dir):
        xml_path = os.path.join(labels_dir, xml_name)
        tree = ET.parse(xml_path)
        plate_text = tree.find('*/platetext').text
        lbls.append(plate_text)

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        imgs.append(img)

    return imgs, lbls


images, labels = load_images_and_labels()
total_count = len(images)
predictions = []
sum_ratio = 0

for idx in range(total_count):
    yolo_predictions = yolo_plate_detection.return_predict(images[idx])
    license_plate_img = crop_plate(images[idx], yolo_predictions)
    license_plate_str = read_plate_characters(license_plate_img)
    predictions.append(license_plate_str)
    print(' Actual:   ', labels[idx])
    print(' Predicted:', license_plate_str)
    ratio = fuzz.ratio(labels[idx], license_plate_str)
    sum_ratio += ratio
    print(' Ratio: ', ratio)
    print('---------------------')
    # cv2.waitKey(0)

score = round(accuracy_score(labels, predictions) * 100, 2)
print('Exactly matching:', score, '%')

ratio_score = round(sum_ratio / total_count, 2)
print('Average ratio:', ratio_score)
