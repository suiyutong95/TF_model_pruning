import tensorflow as tf


####################################################################################
# GPU UTILS
def average_gradients(tower_grads):
    average_grads = []
    print('Gradients Un-available Variables:')
    for grad_and_vars in zip(*tower_grads):
        grads = []
        if grad_and_vars[0][0] == None:
            print('-', grad_and_vars[0][1])
            continue
        # print('**',grad_and_vars[0][1])
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
          'MutableHashTableOfTensors', 'MutableDenseHashTable']
def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            # return"/"+ps_device
            return ps_device
        else:
            return device

    return _assign


def get_available_gpus():
    '''Returns a list of the identifiers of all visible GPUs'''
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    # print('ALL DEVICES:',[x.nameforxinlocal_device_protos])
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


####################################################################################
# TRAINUTILS

def gradient_clip(grads, clip_norm=5, mode='LOCAL'):
    '''
    {clip gradients for tf-format-tensors}
    Params:
        grads[tf.tensor]:
        clip_norm[int/float]: norm threshold
        mode[str]:
    Returns:
        clipped_grads[tf.tensor]:
    '''

    if mode == 'LOCAL':
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, clip_norm), v)
        return grads
    elif mode == 'GLOBAL':
        grads, variables = zip(*grads)
        grads, global_norm = tf.clip_by_global_norm(grads, clip_norm)
        return zip(grads, variables)
    else:
        raise NameError('Clip mode not supported. Only "LOCAL" or "GLOBAL" is supported')
