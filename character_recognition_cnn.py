from tensorflow.keras import layers, models
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime

width = 75
height = 100
channel = 1


def load_data():
    images = np.array([]).reshape(0, height, width)
    labels = np.array([])

    dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10,
                  'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
                  'L': 21, 'M': 22, 'N': 23, 'P': 24, 'Q': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29, 'V': 30,
                  'W': 31, 'X': 32, 'Y': 33, 'Z': 34}

    directories = [directory for directory in glob.glob('datasets/BelgianLicencePlates/TrainLetters/*')]
    for directory in directories:
        file_list = glob.glob(directory + '/*.jpg')
        sub_images = np.array([np.array(Image.open(file_name)) for file_name in file_list])
        sub_labels = [dictionary[directory[-1]]] * len(sub_images)
        images = np.append(images, sub_images, axis=0)
        labels = np.append(labels, sub_labels, axis=0)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)
    return (x_train, y_train), (x_test, y_test)


(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images.reshape((train_images.shape[0], height, width, channel))
test_images = test_images.reshape((test_images.shape[0], height, width, channel))
train_images, test_images = train_images / 255.0, test_images / 255.0
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channel)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(35, activation='softmax'))
start = datetime.now().replace(microsecond=0)
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=8)
end = datetime.now().replace(microsecond=0)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)
print('Test loss: ', test_loss)
print('Training duration: ', (end - start))
model.save('models/character_recognition_cnn.h5')
print('> Saved model to disk <')
