import random
import tensorflow as tf

def crop(image, IMG_X_SIZE, IMG_Y_SIZE):
    x = random.randint(5,15)
    y = random.randint(5,15)
    image = tf.image.crop_to_bounding_box(image=image,
                                          offset_height=y,
                                          offset_width=x,
                                          target_height=IMG_Y_SIZE-y,
                                          target_width=IMG_X_SIZE-x)
    return tf.image.resize(image, [IMG_Y_SIZE,IMG_X_SIZE])

def flip(image):
    image = tf.image.flip_left_right(image)
    return image
