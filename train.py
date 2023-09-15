from os import listdir
from cv2 import cv2
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, Input
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

raw_folder = "Data/"


def save_data(raw_folder):
    images = []
    labels = []

    for folder in listdir(raw_folder):
        if folder != ".DS_Store":
            print("Folder: ", folder)

            for file in listdir(raw_folder + folder):
                if file != ".DS_Store":
                    print("File: ", file)

                    image = cv2.resize(cv2.imread(r"{}{}{}{}".format(raw_folder, folder, "/", file)), dsize=(128, 128))
                    images.append(image)
                    labels.append(folder)

    X = np.array(images)
    Y = np.array(labels)

    encoder = LabelBinarizer()
    Y_onehot = encoder.fit_transform(Y)
    file = open("train.data", "wb")
    pickle.dump((X, Y_onehot), file)
    file.close()


def load_data():
    file = open("train.data", "rb")

    (images, labels) = pickle.load(file)
    file.close()
    print(images.shape)
    print(labels.shape)

    return images, labels


def get_model():
    model = VGG16(weights="imagenet", include_top=False)

    for layer in model.layers:
        layer.trainable = False

    input = Input(shape=(128, 128, 3), name="input_image")
    output = model(input)

    layer = Flatten(name="Flatten")(output)
    layer = Dense(4096, activation='relu', name='fc1')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(4096, activation='relu', name='fc2')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(31, activation='softmax', name='classifier')(layer)

    final_model = Model(inputs=input, outputs=layer)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return final_model


if __name__ == '__main__':

    # Save data to train_data file.
    # save_data(raw_folder)

    """
    Phần sau train model trên Google Colab
    """
    (x_data, y_data) = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=100)

    model = get_model()
    file_path = "weight--{epoch:.02d}--{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # avg = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
    #                          rescale=1. / 255, width_shift_range=0.1,
    #                          height_shift_range=0.1, horizontal_flip=True,
    #                          brightness_range=[0.2, 1.5], fill_mode='nearest')

    my_model = model.fit(x_train, y_train, batch_size=64,
                         epochs=10,
                         validation_data=(x_test, y_test),
                         callbacks=callbacks_list)

    model.save("NHANDANGSOXE.h5")
