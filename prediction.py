import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import tensorflow as tf
import uuid

""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results/prediction")

    """ Load the model """
    model_path = os.path.join("files", "model.h5")
    model = tf.keras.models.load_model(model_path)

    """ Specify the path to the single image you want to process """
    image_path = "test/smile.jpeg"  # Change this to the path of your image

    """ Extracting the name """
    name = os.path.basename(image_path)

    """ Reading the image """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    x = cv2.resize(image, (W, H))
    x = x/255.0
    x = np.expand_dims(x, axis=0)

    """ Prediction """
    pred = model.predict(x, verbose=0)

    """ Save final prediction with removed background as PNG """
    image_h, image_w, _ = image.shape

    mask = pred[0][0]
    mask = cv2.resize(mask, (image_w, image_h))
    mask = np.expand_dims(mask, axis=-1)

    # Create RGBA image with transparency
    result_image = np.concatenate([image, mask * 255], axis=-1).astype(np.uint8)

    # Save as PNG to preserve transparency
    save_image_path = os.path.join("results", "prediction", f"{uuid.uuid1()}.png")
    cv2.imwrite(save_image_path, result_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])