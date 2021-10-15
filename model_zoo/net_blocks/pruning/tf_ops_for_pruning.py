import tensorflow as tf
from pruning_utils import pruning_wrapper

@pruning_wrapper(ch_op_type=None)
def tf_layers_max_pooling3d(*args, **kwargs):
    _block_scope = kwargs.get('_block_scope','tf_layers_max_pooling3d')
    if '_block_scope' in kwargs.keys(): del kwargs['_block_scope']
    with tf.variable_scope(None, _block_scope,):
        return tf.layers.max_pooling3d(*args, **kwargs)

@pruning_wrapper(ch_op_type='concat')
def tf_concat(clist, *args, **kwargs):
    _block_scope = kwargs.get('_block_scope','tf_concat')
    if '_block_scope' in kwargs.keys(): del kwargs['_block_scope']
    with tf.variable_scope(None, _block_scope,):
        return tf.concat(clist, *args, **kwargs)

@pruning_wrapper(ch_op_type=None)
def tf_nn_softmax(*args, **kwargs):
    _block_scope = kwargs.get('_block_scope','tf_nn_softmax')
    if '_block_scope' in kwargs.keys(): del kwargs['_block_scope']
    with tf.variable_scope(None, _block_scope,):
        return tf.nn.softmax(*args, **kwargs)

@pruning_wrapper(ch_op_type=None)
def tf_layers_dropout(*args, **kwargs):
    _block_scope = kwargs.get('_block_scope','tf_layers_dropout')
    if '_block_scope' in kwargs.keys(): del kwargs['_block_scope']
    with tf.variable_scope(None, _block_scope,):
        return tf.layers.dropout(*args, **kwargs)

@pruning_wrapper(ch_op_type=None)
def tf_identity(*args, **kwargs):
    _block_scope = kwargs.get('_block_scope', 'tf_identity')
    if '_block_scope' in kwargs.keys(): del kwargs['_block_scope']
    with tf.variable_scope(None, _block_scope,):
        return tf.identity(*args, **kwargs)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> name map >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
_function_name_map = {
    'tf_layers_max_pooling3d': tf_layers_max_pooling3d,
    'tf_concat': tf_concat,
    'tf_nn_softmax': tf_nn_softmax,
    'tf_layers_dropout': tf_layers_dropout,
    'tf_identity': tf_identity,
}