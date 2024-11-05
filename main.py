import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# بارگذاری تصویر
def load_and_preprocess_image(file_path):
    img = load_img(file_path, target_size=(150, 150))  # اندازه تصویر
    img_array = img_to_array(img) / 255.0  # نرمال‌سازی
    return img_array

# بارگذاری داده‌ها
cat_images = [load_and_preprocess_image(f'/content/kagglecatsanddogs_3367a/PetImages/Cat/{filename}') for filename in os.listdir('/content/kagglecatsanddogs_3367a/PetImages/Cat')]
dog_images = [load_and_preprocess_image(f'/content/kagglecatsanddogs_3367a/PetImages/Dog/{filename}') for filename in os.listdir('/content/kagglecatsanddogs_3367a/PetImages/Dog')]

X = np.array(cat_images + dog_images)
y = np.array([0] * len(cat_images) + [1] * len(dog_images))  # 0 برای گربه، 1 برای سگ
y = to_categorical(y, num_classes=2)

# ساخت مدل CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# آموزش مدل
model.fit(X, y, epochs=5, batch_size=32)

# ذخیره مدل
model.save('cat_dog_classifier.h5')
