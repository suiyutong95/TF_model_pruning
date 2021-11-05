import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops

# from model_zoo.pruning_utils import pruning_wrapper

_INTEGRATE_MODE = False

'''7x7x7 conv for starting network'''
def start_77conv_block(x, channels, norm='GN', active_func='Relu'):
    x = tf.layers.conv3d(x, filters=channels, strides=1, kernel_size=7, padding='SAME')
    x = _norm_func(x, norm)
    x = _active_func(x, active_type=active_func)
    return x


# def conv3D_block (x, channels , norm='GN' ,activ_func='Relu'):
#     '''3d conv block'''
#     x = tf.layers.conv3d(x, filters=channels, strides=1, kernel_size=3, padding=' SAME ')
#     # Cannot use '_ norm func' here due to it use 'tf. contrib. layers.batch norm '
#     # instead of 'tf. layers. batch normalization' in '_ norm func' ,
#     # which is all the same but has a different 'name' of tensor.
#     # This might casue name error when reload old-version ckpts.
#     # can change to ' norm func' in future when donot need load old-version ckpts
#     if norm  == 'GN' :
#         if _INTEGRATE_MODE:
#             x = group_norm2(x, groups=4)
#             print( 'INTEGRATE MODE IS ON')
#         else:
#             x = tf.contrib.layers.group_norm(x, groups=4, reduction_axes=(-4,-3,-2))
#     elif norm == 'BN':
#         x = tf.contrib.layers.batch_norm(x)
#     x = _active_func(x,acticv_type=activ_func)
#
#     return x

'''3d conv block'''
def conv3D_block(x, channels, norm='GN', active_func='Relu'):
    '''
    3d conv block
    '''
    x = tf.layers.conv3d(x, filters=channels, strides=1, kernel_size=3, padding='SAME')
    x = _norm_func(x, norm_type=norm)
    x = _active_func(x, active_type=active_func)
    return x


'''SE block'''
def _GAP(x):
    return tf.reduce_mean(x, axis=[1, 2, 3])
def SE_block_3d(x, ratio=8, active_func='Relu'):
    w = _GAP(x)
    channels = w.get_shape()[-1]
    if ratio > channels:
        ratio = channels
    w = tf.layers.dense(w, units=channels//ratio, )
    w = _active_func(w, active_type=active_func)
    w = tf.layers.dense(w, units=channels, )
    w = tf.nn.sigmoid(w)
    w = tf.reshape(w, [-1, 1, 1, 1, channels])
    return x*w


'''SE-Conv block'''
def conv3D_SE_block(x, channels, norm='GN', active_func='Relu'):
    x = conv3D_block(x, channels, norm=norm, active_func=active_func)
    x = SE_block_3d(x, ratio=8, active_func=active_func)
    return x


'''Res block'''
def Res_block_3d(inputs, size=3, norm='BN', active_func='Relu'):
    channels = inputs.get_shape().as_list()[-1]
    x = tf.layers.conv3d(
        inputs=inputs, filters=channels, kernel_size=size, strides=1, padding='SAME')
    x = _norm_func(x, norm)
    x = _active_func(x, active_type=active_func)
    x = tf.layers.conv3d(
        inputs=x, filters=channels, kernel_size=size, strides=1, padding='SAME ')
    x = _norm_func(x, norm)
    x = inputs+x
    x = _active_func(x, active_type=active_func)
    return x


'''ResX block '''
def _group_conv3d(x, groups=16, channels=128, size=3):
    subchannel = channels/groups
    input_list = tf.split(x, num_or_size_splits=groups, axis=-1)
    output_list = []
    for x in input_list:
        output_list.append(
            tf.layers.conv3d(x, filters=subchannel, kernel_size=size, strides=1, padding='SAME')
        )
    res = tf.concat(output_list, axis=-1)
    return res
def ResX_block_3d(inputs, mid_channels=None, size=3, groups=32, norm='BN', active_func='Relu'):
    channels = int(inputs.get_shape()[-1])
    if not mid_channels:
        mid_channels = channels//2
    if mid_channels < 4:
        norm = 'BN'
    x = tf.layers.conv3d(
        inputs=inputs, filters=mid_channels, kernel_size=1, strides=1, padding='SAME')
    x = _norm_func(x, norm)
    x = _active_func(x, active_type=active_func)
    x = _group_conv3d(x, groups=groups, channels=mid_channels, size=size)
    x = _norm_func(x, norm)
    x = tf.layers.conv3d(
        inputs=x, filters=channels, kernel_size=1, strides=1, padding='SAME')
    x = _norm_func(x, norm)
    x = inputs+x
    x = _active_func(x, active_type=active_func)
    return x


'''SE-Res block'''
def SE_Res_block_3d(x, ratio=16, norm='GN', active_func='Relu'):
    x = Res_block_3d(x, norm=norm, active_func=active_func)
    x = SE_block_3d(x, ratio=ratio, active_func=active_func)
    return x


'' 'SE-ResX block'' '
def SE_ResX_block_3d(x, mid_channels=None, opt_channels=None,
                     groups=16, ratio=16, norm='GN', active_func='Relu'):
    x = ResX_block_3d(
        x, mid_channels=mid_channels, groups=groups, norm=norm, active_func=active_func)
    x = SE_block_3d(x, ratio=ratio, active_func=active_func)
    return x


'''Pseudo 3d Conv'''
def pseudo3d_conv_Res_block(x, channels, norm='GN', active_func='Relu'):
    _sub_ch = channels//2
    x_res = x
    x_Me = tf.layers.conv3d(x, filters=_sub_ch, strides=1, kernel_size=[1, 3, 3], padding='SAME')
    x_Me = _norm_func(x_Me, norm)
    x_Me = _active_func(x_Me, active_type=active_func)
    x_Co = tf.layers.conv3d(x, filters=_sub_ch, strides=1, kernel_size=[3, 1, 3], padding='SAME')
    x_Co = _norm_func(x_Co, norm)
    x_Co = _active_func(x_Co, active_type=active_func)
    x_Tr = tf.layers.conv3d(x, filters=_sub_ch, strides=1, kernel_size=[3, 3, 1], padding='SAME')
    x_Tr = _norm_func(x_Tr, norm)
    x_Tr = _active_func(x_Tr, active_type=active_func)
    x = x_Me+x_Co+x_Tr
    x = tf.layers.conv3d(x, filters=channels, strides=1, kernel_size=1, padding='SAME')
    x = _norm_func(x, norm)
    X = x_res+x
    x = _active_func(x, active_type=active_func)
    return x


############################################## up samples ##############################################

'''3d conv-transpose block'''
def conv3D_T_block(x, channels, norm='GN', active_func='Relu', upsampleDims='ALL'):
    if upsampleDims == 'ALL':
        x = tf.layers.conv3d_transpose(x, filters=channels, strides=2, kernel_size=4, padding='SAME')
    elif upsampleDims == 'NoDepth':
        x = tf.layers.conv3d_transpose(x, filters=channels, strides=[1, 2, 2], size=4, padding='SAME')
    else:
        raise NameError('Undifined upsample type')
    x = _norm_func(x, norm)
    x = _active_func(x, active_type=active_func)
    return x


'''3d upsample block'''
def upsample3D_block(x, channels, norm='GN', active_func='Relu',
                     upsampleDims='ALL', upsampleFactor=None):
    if upsampleDims == 'ALL':
        x = _resize_3D(x, scale_factor=[2, 2, 2])
    elif upsampleDims == 'NoDepth':
        x = _resize_3D(x, scale_factor=[1, 2, 2])
    elif upsampleDims == 'USE_ INPUT':
        x = _resize_3D(x, scale_factor=upsampleFactor)
    else:
        raise NameError('Undifined upsample type ')
    x = tf.layers.conv3d(x, filters=channels, strides=1, kernel_size=3, padding='SAME')
    x = _norm_func(x, norm)
    x = _active_func(x, active_type=active_func)
    return x


####################################################################################################################################################
def group_norm2(inputs, groups=32, ):
    channels_axis = 4
    dyanmic_shape = tf.shape(inputs)
    input_shape_list = []
    for i, dim in enumerate(inputs.shape):
        if dim.value is None:
            input_shape_list.append(dyanmic_shape[i])
        else:
            input_shape_list.append(dim)
    reduction_axes = [1, 2, 3]
    channels = int(inputs.shape[-1])
    params_shape_broadcast = [1, 1, 1, 1, groups, channels//groups]
    inputs_shape = input_shape_list[:channels_axis]+[groups, channels//groups]
    inputs = tf.reshape(inputs, inputs_shape)
    moments_axes = [5, 1, 2, 3]
    with variable_scope.variable_scope(
            None, 'GroupNorm', [inputs], reuse=None) as sc:
        params_shape = [channels]

        beta = tf.get_variable('beta', [channels], initializer=tf.constant_initializer(1.0))
        beta = tf.reshape(beta, params_shape_broadcast)

        gamma = tf.get_variable('gamma', [channels], initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, params_shape_broadcast)

        mean, variance = tf.nn.moments(inputs, moments_axes, keep_dims=True)

        gain = math_ops.rsqrt(variance+1e-6)
        offset = -mean*gain
        gain *= gamma
        offset *= gamma
        offset += beta
        outputs = inputs*gain+offset

        outputs = tf.reshape(outputs, input_shape_list)

        return outputs


####################################################################################################################################################
def _active_func(x, active_type):
    if active_type == 'Relu':
        return tf.nn.relu(x)
    elif active_type == 'LeakyRelu':
        return tf.nn.leaky_relu(x)
    else:
        raise NameError('Un-defined Activation Function')


def _norm_func(x, norm_type):
    if norm_type == 'BN':
        return tf.layers.batch_normalization(x)
    elif norm_type == 'GN':
        if _INTEGRATE_MODE:
            return group_norm2(x, groups=4)
        else:
            return tf.contrib.layers.group_norm(x, groups=4, reduction_axes=(-4, -3, -2))
    else:
        raise NameError('Undifined normalization type')


def _resize_3D(input_layer, scale_factor):
    _method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    shape = input_layer.get_shape()
    x = tf.reshape(input_layer, [-1, shape[1], shape[2], shape[3]*shape[4]])
    x = tf.image.resize_images(x, [shape[1]*scale_factor[0], shape[2]*scale_factor[1]], _method)
    x = tf.reshape(x, [-1, shape[1]*scale_factor[0], shape[2]*scale_factor[1], shape[3], shape[4]])
    x = tf.transpose(x, [0, 3, 2, 1, 4])
    x = tf.reshape(x, [-1, shape[3], shape[2]*scale_factor[1], shape[1]*scale_factor[0]*shape[4]])
    x = tf.image.resize_images(x, [shape[3]*scale_factor[2], shape[2]*scale_factor[1]], _method)
    x = tf.reshape(x, [-1, shape[3]*scale_factor[2], shape[2]*scale_factor[1], shape[1]*scale_factor[0], shape[4]])
    x = tf.transpose(x, [0, 3, 2, 1, 4])
    return x
