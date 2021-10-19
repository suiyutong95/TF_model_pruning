import tensorflow as tf

from pruning_utils import pruning_wrapper

'''
net-bloocks which support pruning
'''

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> basic functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

'''
[*keep '_block_scope' same as function name*]
'''

'''7x7x7 conv for starting network'''
@pruning_wrapper(allow_prune=True)
def start_77conv_block(x, channels, norm='GN', active_func='Relu', _block_scope='start_77conv_block'):
    '''
    {*keep '_block_scope' same as function name*}
    '''
    with tf.variable_scope(None, _block_scope, ):
        x = tf.layers.conv3d(x, filters=channels, strides=1, kernel_size=7, padding='SAME')
        x = _norm_func(x, norm)
        x = _active_func(x, active_type=active_func)
        return x


'''3d conv block'''
@pruning_wrapper(allow_prune=True)
def conv3D_block(x, channels, norm='GN', active_func='Relu', _block_scope='conv3D_block'):
    with tf.variable_scope(None, _block_scope, ):
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
@pruning_wrapper()
def SE_block_3d(x, ratio=8, active_func='Relu', _block_scope='SE_block_3d'):
    with tf.variable_scope(None, _block_scope, ):
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


'''3d upsample block'''
@pruning_wrapper()
def upsample3D_block(x, channels, norm='GN', active_func='Relu',
                     upsampleDims='ALL', upsampleFactor=None, _block_scope='upsample3D_block'):
    with tf.variable_scope(None, _block_scope, ):
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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> second level functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

'''SE-Conv block'''
def conv3D_SE_block(x, channels, norm='GN', active_func='Relu'):
    x = conv3D_block(x, channels, norm=norm, active_func=active_func)
    x = SE_block_3d(x, ratio=8, active_func=active_func)
    return x


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> private functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def _active_func(x, active_type):
    if active_type == 'Relu':
        return tf.nn.relu(x)
    elif active_type == 'LeakyRelu':
        return tf.nn.leaky_relu(x)
    else:
        raise NameError('Un-defined Activation Function')


def _norm_func(x, norm_type):
    if norm_type == 'BN':
        return tf.contrib.layers.batch_norm(x,scale=True)
    elif norm_type == 'GN':
        return tf.contrib.layers.group_norm(x, groups=4, reduction_axes=(-4, -3, -2))
    else:
        raise NameError('Un-defined normalization type')


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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> name map >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
_function_name_map = {
    'start_77conv_block': start_77conv_block,
    'conv3D_block': conv3D_block,
    'SE_block_3d': SE_block_3d,
    'upsample3D_block': upsample3D_block,
}
