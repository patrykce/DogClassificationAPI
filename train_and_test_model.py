import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
import time
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NAME = f'Dog_Breed_Recognition_{int(time.time())}'
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


def build_model(size, num_classes):
    inputs = Input((size, size, 3))
    backbone = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")
    backbone.trainable = True
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, x)
    return model


def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def parse_data(x, y):
    x = x.decode()

    num_class = 120
    size = 224

    image = read_image(x, size)
    label = [0] * num_class
    label[y] = 1
    label = np.array(label)
    label = label.astype(np.int32)

    return image, label


def tf_parse(x, y):
    x, y = tf.numpy_function(parse_data, [x, y], [tf.float32, tf.int32])
    x.set_shape((224, 224, 3))
    y.set_shape((120))
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


if __name__ == "__main__":
    path = "Dog Breed Identification/"
    train_path = os.path.join(path, "train/*")
    test_path = os.path.join(path, "test/*")
    labels_path = os.path.join(path, "labels.csv")

    csv_labels_data = pd.read_csv(labels_path)
    breeds = csv_labels_data["breed"].unique()

    enumerated_breeds = {name: i for i, name in enumerate(breeds)}

    ids = glob(train_path)
    labels = []

    for image_id in ids:
        image_id = image_id.split("/")[-1].split(".")[0]
        breed_name = list(csv_labels_data[csv_labels_data.id == image_id]["breed"])[0]
        final_breed_label = enumerated_breeds[breed_name]
        labels.append(final_breed_label)

    train_ids = ids[:30]
    test_ids = ids[10000:]
    train_labels = labels[:30]
    test_labels = labels[10000:]

    train_x, valid_x = train_test_split(train_ids, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(train_labels, test_size=0.2, random_state=42)

    size = 224
    num_classes = 120
    lr = 1e-4
    batch = 16
    epochs = 13

    model = build_model(size, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["acc"])

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    callbacks = [
        ModelCheckpoint("t_v1.h5", verbose=1, save_best_only=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
        tensorboard
    ]
    train_steps = (len(train_x) // batch) + 1
    valid_steps = (len(valid_x) // batch) + 1

    history = model.fit(train_dataset,
                        steps_per_epoch=train_steps,
                        validation_data=valid_dataset,
                        validation_steps=valid_steps,
                        epochs=epochs,
                        callbacks=callbacks)


    def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error [Breed classification]')
        plt.legend()
        plt.grid(True)
        plt.show()


    plot_loss(history)

    _, test_x = train_test_split(test_ids, test_size=0.5, random_state=42)
    _, test_y = train_test_split(test_labels, test_size=0.5, random_state=42)
    train_steps = (len(test_x) // batch) + 1
    test_dataset = tf_dataset(test_x, test_y, batch=batch)
    model.evaluate(test_dataset, batch_size=batch, steps=train_steps, verbose=2, callbacks=tensorboard)
