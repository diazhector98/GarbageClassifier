import cv2
import tensorflow as tf
import os

CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
TESTING_DATA_DIR = "testing_data"

def prepare(file):
    IMG_SIZE = 150
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")

for category in CATEGORIES:
    path = os.path.join(TESTING_DATA_DIR, category)
    image_directories = os.listdir(path)
    images_correctly_classified = 0
    images_incorrectly_classified = 0
    images_dictionary = {"cardboard":0, "glass":0, "metal":0, "paper":0, "plastic":0, "trash": 0}
    for image in image_directories:
        prediction = model.predict(prepare(os.path.join(path, image)))
        prediction = list(prediction[0])
        best_prediction = CATEGORIES[prediction.index(max(prediction))]
        images_dictionary[best_prediction] += 1
        if best_prediction == category:
            images_correctly_classified += 1
        else:
            images_incorrectly_classified += 1
    print("Category: " + category)
    print("Accuracy: " + str(images_correctly_classified / (images_incorrectly_classified + images_correctly_classified)))
    print(images_dictionary)
# image = "test.jpg" #your image path
# prediction = model.predict([image])
# prediction = list(prediction[0])
# print(CATEGORIES[prediction.index(max(prediction))])
