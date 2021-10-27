import os
import sys
# sys.path.append(os.path.abspath('../'))

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import json

from utils.system import get_cpu_nums
from ..image_utils.CTA_image import CTA_norm_tf
from ..image_utils.transform import random_rotate_l_r_3, random_flip_l_r_3, random_zoom_xy, \
    pickmid_ingraph, pickrandom2


def _parse_example(example_proto):
    '''a simple pasre function'''
    # map_func: apply to each element of this dataset
    features = {
        'img_id': tf.FixedLenFeature([], tf.string),
        'img_size': tf.FixedLenFeature([], tf.string),
        'img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
        'vessel': tf.FixedLenFeature([], tf.string),
        'heart': tf.FixedLenFeature([], tf.string),
        'bbox_size': tf.FixedLenFeature([], tf.string),
        'bbox': tf.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.parse_single_example(example_proto, features)

    image_size = tf.decode_raw(parsed_example['img_size'], tf.int32)
    image_size.set_shape([3])

    image_id = parsed_example['img_id']

    # 3-dimention
    image = tf.decode_raw(parsed_example['img'], tf.float32)
    image = tf.reshape(image, image_size)
    image = CTA_norm_tf(image)

    label = tf.decode_raw(parsed_example['label'], tf.int8)
    label = tf.reshape(label, image_size)
    label = tf.cast(label, tf.float32)

    vessel = tf.decode_raw(parsed_example['vessel'], tf.int8)
    vessel = tf.reshape(vessel, image_size)
    vessel = tf.cast(vessel, tf.float32)
    vessel = tf.clip_by_value(vessel, 0, 1)

    heart = tf.decode_raw(parsed_example['heart'], tf.int8)
    heart = tf.reshape(heart, image_size)
    heart = tf.cast(heart, tf.float32)

    return {
        'image_size': image_size,
        'image_id': image_id,
        'image': image,
        'label': label,
        'vessel': vessel,
        'heart': heart,
    }


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> light augs >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def _lightaug_randomflip(example_dict):
    image, label, vessel, heart = \
        example_dict['image'], example_dict['label'], example_dict['vessel'], example_dict['heart']
    image, label, vessel, heart = random_flip_l_r_3([image, label, vessel, heart])
    example_dict['image'], example_dict['label'], example_dict['vessel'], example_dict[
        'heart'] = image, label, vessel, heart
    return example_dict


def _lightaug_HUalter(example_dict):
    image = example_dict['image']
    alter = tf.random_normal([], mean=0, stddev=4, dtype=tf.int32)
    image = image+alter
    example_dict['image'] = image
    return example_dict


def _lightaug_POWalter(example_dict):
    image = example_dict['image']
    alter = tf.random_normal([], mean=1, stddev=.03, dtype=tf.float32)
    image = tf.pow(image, alter)
    example_dict['image'] = image
    return example_dict


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> heave augs >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def _heavyaug_randomrotate(example_dict):
    image, label, vessel, heart = \
        example_dict['image'], example_dict['label'], example_dict['vessel'], example_dict['heart']
    image, label, vessel, heart = random_rotate_l_r_3([image, label, vessel, heart])
    example_dict['image'], example_dict['label'], example_dict['vessel'], example_dict['heart'] = \
        image, label, vessel, heart
    return example_dict


def _heavyaug_randomzoom(example_dict):
    image, label, vessel, heart = \
        example_dict['image'], example_dict['label'], example_dict['vessel'], example_dict['heart']
    image, label, vessel, heart = random_zoom_xy([image, label, vessel, heart])
    example_dict['image'], example_dict['label'], example_dict['vessel'], example_dict['heart'] = \
        image, label, vessel, heart
    return example_dict


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> crop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def _get_crop(slice_depth, slice_size, crop_method):
    if crop_method == 'pickmid':
        def _parse_func(example_dict, ):
            image, label, vessel, heart, image_size = \
                example_dict['image'], example_dict['label'], example_dict['vessel'], \
                example_dict['heart'], example_dict['image_size']
            image = pickmid_ingraph(image, image_size, depth=slice_depth, size=slice_size)
            label = pickmid_ingraph(label, image_size, depth=slice_depth, size=slice_size)
            vessel = pickmid_ingraph(vessel, image_size, depth=slice_depth, size=slice_size)
            heart = pickmid_ingraph(heart, image_size, depth=slice_depth, size=slice_size)
            example_dict['image'], example_dict['label'], example_dict['vessel'], example_dict['heart'] = \
                image, label, vessel, heart
            return example_dict

    elif crop_method == 'random':
        def _parse_func(example_dict, ):
            image, label, vessel, heart, image_size = \
                example_dict['image'], example_dict['label'], example_dict['vessel'], \
                example_dict['heart'], example_dict['image_size']
            image, label, vessel, heart = pickrandom2(
                [image, label, vessel, heart], image_size, depth=slice_depth, size=slice_size)
            example_dict['image'], example_dict['label'], example_dict['vessel'], example_dict['heart'] = \
                image, label, vessel, heart
            return example_dict
    else:
        assert 0

    return _parse_func


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def make_batch_iterator_V3(tfrecord_path, is_train=True, aug_on=False, batch_size=4,
                           parse_func='discarded-param', slice_depth=8, slice_size=384):
    """
    {}
    Input:
        tfrecord_path[str]:
        is_train[bool]:
        batch_size[int]:
        parse_func[str]:choosablefrom['random','pickmid']
    Output:
    """
    _cpus = min(8, get_cpu_nums())
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    if is_train:
        dataset = dataset.shuffle(100)  # shuffle datas
        dataset = dataset.repeat(10000)
    # parse example
    dataset = dataset.map(_parse_example, num_parallel_calls=_cpus)
    # DO AUGMENTATION
    if is_train and aug_on:
        # heavy aug
        dataset = dataset.map(_heavyaug_randomzoom, num_parallel_calls=_cpus)
        dataset = dataset.map(_heavyaug_randomrotate, numparallel_calls=_cpus)

    # data echo
    dataset = dataset.flat_map(lambda t: tf.data.Dataset.from_tensors(t).repeat(2))

    if is_train and aug_on:
        # light aug
        dataset = dataset.map(_lightaug_randomflip, num_parallel_calls=_cpus)
        dataset = dataset.map(_lightaug_POWalter, num_parallel_calls=_cpus)

    # crop
    if is_train:
        dataset = dataset.map(
            _get_crop(slice_depth, slice_size, crop_method='random'), num_parallel_calls=_cpus
        )
    else:
        dataset = dataset.map(
            _get_crop(slice_depth, slice_size, crop_method='pickmid'), num_parallel_calls=_cpus
        )

    # add batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # pre-load examples
    _autotune = False
    if _autotune:
        dataset = dataset.prefetch(-1)
    else:
        dataset = dataset.prefetch(4*batch_size)

    return dataset.make_initializable_iterator()
