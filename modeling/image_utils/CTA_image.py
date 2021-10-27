import tensorflow as tf


def CTA_norm(image):
    image[image < -1024] = -1024
    image[image > 2048] = 2048
    image_norm = (image+1024)/3072
    return image_norm


def CTA_norm_tf(image):
    image_clip = tf.clip_by_value(image, -1024, 2048)
    image_norm = (image_clip+1024)/3072
    return image_norm
