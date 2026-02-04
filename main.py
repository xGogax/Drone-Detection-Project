import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras import layers

from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.models import load_model

#Treniranje konvolucione mreze
def cnn_model(num_classes, data_augmentation):
    model = Sequential([
        data_augmentation,

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def main():
    train_path = "./data/train"
    valid_path = "./data/valid"
    img_size = (64, 64)
    batch_size = 128

    #Inicijalizacija ulaznih podataka
    Xtrain = image_dataset_from_directory(train_path,
                                          image_size=img_size,
                                          batch_size=batch_size,
                                          shuffle=True)

    Xvalid = image_dataset_from_directory(valid_path,
                                          image_size=img_size,
                                          batch_size=batch_size,
                                          shuffle=True)

    #Klasifikacija podataka
    classes = Xtrain.class_names
    num_classes = len(classes)
    print("Dataset ima sledece klase ", classes)

    #Prikaz podataka
    img, lab = next(iter(Xtrain))
    plt.figure(figsize=(16, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')
    plt.show()

    #Augmentacija podataka
    data_augmentation = Sequential(
        [
            layers.Input((img_size[0], img_size[1], 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ]
    )
    for i in range(5):
        aug_img = data_augmentation(img)
        plt.subplot(2, 5, i + 1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')
    plt.show()

    #Model
    model = cnn_model(num_classes, data_augmentation)
    model.summary()

    #EarlyStopping
    es = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
    history = model.fit(Xtrain,
                        epochs=15,
                        validation_data=Xvalid,
                        callbacks=[es],
                        verbose=1)

    #Plotovanje metrike
    plt.figure()
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()