import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import tensorflow as tf

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
    for item in ["prediction", "joint"]:
        create_dir(f"results/{item}")

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

    line = np.ones((H, 10, 3)) * 255

    """ Joint and save mask in a folder results/masks """
    '''
    pred_list = []
    for item in pred:
        p = item[0] * 255
        p = np.concatenate([p, p, p], axis=-1)

        pred_list.append(p)
        pred_list.append(line)

    save_image_path = os.path.join("results", "mask", name)
    cat_images = np.concatenate(pred_list, axis=1)
    cv2.imwrite(save_image_path, cat_images)
    '''

    """ Save final mask """
    image_h, image_w, _ = image.shape

    y0 = pred[0][0]
    y0 = cv2.resize(y0, (image_w, image_h))
    y0 = np.expand_dims(y0, axis=-1)
    y0 = np.concatenate([y0, y0, y0], axis=-1)

    line = line = np.ones((image_h, 10, 3)) * 255

    cat_images = np.concatenate([image, line, y0*255, line, image*y0], axis=1)
    save_image_path = os.path.join("results", "joint", name)
    cv2.imwrite(save_image_path, cat_images)
    print("saved the images with mask successfuly")