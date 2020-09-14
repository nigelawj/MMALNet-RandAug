import tensorflow as tf
import numpy as np
import pandas as pd
import random
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from pathlib import Path
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from skimage.io import imread
from skimage.transform import resize

def get_pic(img_path):
    return np.array(Image.open(img_path).convert('RGB').resize((256,256),Image.ANTIALIAS))

def shuffle_together(*lists):
    # randomly shuffles lists together
    # e.g. [a, b, c], [1, 2, 3] => [c, a, b], [3, 1, 2]
    temp = list(zip(*lists))
    random.shuffle(temp)
    return [list(tup) for tup in zip(*temp)]

df = pd.read_csv("data.csv")
df_non_test = df[df["train"]==True].reset_index(drop=True)
df_test = df[df["train"]==False].reset_index(drop=True)

NUM_CLASSES = df["make_code"].nunique()

IMG_SIZE = 224
BATCH_SIZE = 64

class ImageGenerator(tf.keras.utils.Sequence) :
    def __init__(self, img_filenames, labels, batch_size, img_shape):
        img_filenames, labels = shuffle_together(img_filenames, labels) 
        self.img_filenames = img_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.img_shape = img_shape
    
    def __len__(self) :
        return (np.ceil(len(self.img_filenames) / self.batch_size)).astype(np.int)
  
    def __getitem__(self, idx) :
        batch_x = self.img_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        image_data = []
        for file_name in batch_x:
            image_data.append(resize(imread(file_name), (self.img_shape, self.img_shape, 3)))
        return np.array(image_data)/255.0, np.array([to_categorical(y_ele, num_classes=NUM_CLASSES) for y_ele in batch_y])

skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(np.zeros(len(df_non_test)), df_non_test["type"]):
    df_train = df_non_test.iloc[train_index]
    df_val = df_non_test.iloc[test_index]
    train_gen = ImageGenerator(df_train["img_path"].tolist(), df_train["make_code"].tolist(), BATCH_SIZE, IMG_SIZE)
    val_gen = ImageGenerator(df_val["img_path"].tolist(), df_val["make_code"].tolist(), BATCH_SIZE, IMG_SIZE)

def get_augmentation():
    return Sequential([
                preprocessing.RandomRotation(factor=0.15),
                preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                preprocessing.RandomFlip(),
                preprocessing.RandomContrast(factor=0.1),
            ],
            name="img_augmentation")

def build_model(num_classes, weights_path, image_size):
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = get_augmentation()(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights=weights_path)

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)
    x = Dropout(0.2, name="top_dropout")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    
    return model

weights_path = str(Path("./ckpt", "noisy_student_efficientnet-b0", "efficientnetb0_notop.h5"))

model = build_model(NUM_CLASSES, weights_path, IMG_SIZE)

model.fit(x=train_gen,
          steps_per_epoch=len(df_train) // BATCH_SIZE,
          epochs=10,
          verbose=1,
          validation_data=val_gen,
          validation_steps=len(df_val) // BATCH_SIZE)