import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils import image_dataset_from_directory
from keras.models import Sequential, load_model
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from keras.applications import MobileNetV2


class DroneClassifier:
    # --- Inicijalizacija klase ---
    def __init__(self, train_path, valid_path, img_size=(64, 64), batch_size=128,
                 model_path="drone_model.keras", history_path="drone_history.npy"):
        self.train_path = train_path
        self.valid_path = valid_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model_path = model_path
        self.history_path = history_path
        self.classes = None
        self.num_classes = None
        self.model = None
        self.data_augmentation = None
        self.history = None

        # Dataset za običan CNN
        self.Xtrain = image_dataset_from_directory(train_path,
                                                   image_size=img_size,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        self.Xvalid = image_dataset_from_directory(valid_path,
                                                   image_size=img_size,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        # Dataset za transfer learning (MobileNetV2)
        self.Xtrain_tl = image_dataset_from_directory(train_path,
                                                      image_size=(224, 224),
                                                      batch_size=batch_size,
                                                      shuffle=True)
        self.Xvalid_tl = image_dataset_from_directory(valid_path,
                                                      image_size=(224, 224),
                                                      batch_size=batch_size,
                                                      shuffle=True)

        self.classes = self.Xtrain.class_names
        self.num_classes = len(self.classes)
        print("Dataset ima sledece klase:", self.classes)

    # --- Prikaz uzoraka ---
    def show_samples(self, n=10):
        img, lab = next(iter(self.Xtrain))
        plt.figure(figsize=(16, 8))
        for i in range(n):
            plt.subplot(2, n // 2, i + 1)
            plt.imshow(img[i].numpy().astype('uint8'))
            plt.title(self.classes[lab[i]])
            plt.axis('off')
        plt.show()

    # --- Kreiranje augmentacije podataka ---
    def create_augmentation(self):
        self.data_augmentation = Sequential([
            layers.Input((self.img_size[0], self.img_size[1], 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ])

    # --- Prikaz augmentovanih slika ---
    def show_augmented(self, n=5):
        if self.data_augmentation is None:
            raise ValueError("create_augmentation() prvo treba da se pozove")
        img, _ = next(iter(self.Xtrain))

        cols = min(n, 5)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(4 * cols, 4 * rows))
        for i in range(n):
            aug_img = self.data_augmentation(img)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(aug_img[0].numpy().astype('uint8'))
            plt.axis('off')
        plt.show()

    # --- Kreiranje CNN modela ---
    def build_model(self):
        if self.data_augmentation is None:
            raise ValueError("create_augmentation() prvo treba da se pozove")

        self.model = Sequential([
            self.data_augmentation,
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
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    # --- Trening modela ---
    def train(self, epochs=15, patience=10):
        if self.model is None:
            self.build_model()

        es = EarlyStopping(monitor='val_accuracy', patience=patience,
                           restore_best_weights=True, verbose=1)
        history = self.model.fit(
            self.Xtrain,
            validation_data=self.Xvalid,
            epochs=epochs,
            callbacks=[es],
            verbose=1
        )
        self.model.save(self.model_path)
        self.history = history.history
        np.save(self.history_path, self.history)
        return history

    # --- Ucitavanje ili treniranje modela ---
    def load_or_train(self, epochs=15, patience=10):
        if os.path.exists(self.model_path):
            print(f"Ucitam model iz {self.model_path} ...")
            self.model = load_model(self.model_path)
            if os.path.exists(self.history_path):
                self.history = np.load(self.history_path, allow_pickle=True).item()
            return None
        else:
            print("Model ne postoji, treniram novi...")
            return self.train(epochs=epochs, patience=patience)

    # --- Prikaz istorije ---
    def plot_history(self, history=None):
        hist = history.history if history is not None else self.history
        if hist is None:
            print("Nema istorije, model je samo ucitan.")
            return
        plt.figure()
        plt.subplot(121)
        plt.plot(hist['accuracy'], label='train')
        plt.plot(hist['val_accuracy'], label='val')
        plt.title('Accuracy')
        plt.legend()
        plt.subplot(122)
        plt.plot(hist['loss'], label='train')
        plt.plot(hist['val_loss'], label='val')
        plt.title('Loss')
        plt.legend()
        plt.show()

    # --- Evaluacija modela po batch-evima ---
    def evaluate_model(self, use_tl=False):
        X = self.Xvalid_tl if use_tl else self.Xvalid

        y_true_list = []
        y_pred_list = []

        for img, lab in X:
            y_true_list.extend(lab.numpy())
            preds = self.model.predict(img, verbose=0)
            y_pred_list.extend(np.argmax(preds, axis=1))

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        print(f"Tačnost modela je: {100 * accuracy_score(y_true, y_pred):.2f}%")

        cm = confusion_matrix(y_true, y_pred, normalize='true')
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes).plot()
        plt.show()

    # --- Transfer learning model ---
    def transfer_learning_model(self):
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        self.model = Sequential([
            layers.Resizing(224, 224),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(
            optimizer='adam',
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )


if __name__ == "__main__":
    classifier = DroneClassifier("./data/train", "./data/valid")

    # --- Prikaz uzoraka ---
    classifier.show_samples()

    # --- Kreiranje augmentacije ---
    classifier.create_augmentation()
    classifier.show_augmented()

    # --- Ucitavanje ili treniranje modela ---
    history = classifier.load_or_train()
    classifier.plot_history(history)

    # --- Evaluacija ---
    classifier.evaluate_model()

    # --- Transfer learning ---
    classifier.transfer_learning_model()
    es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    history_tl = classifier.model.fit(
        classifier.Xtrain_tl,
        validation_data=classifier.Xvalid_tl,
        epochs=30,
        callbacks=[es],
        verbose=1
    )
    classifier.plot_history(history_tl)
    classifier.evaluate_model(use_tl=True)