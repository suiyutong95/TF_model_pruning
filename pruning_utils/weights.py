import tensorflow as tf


def load_weights_2_tfsess(weight_dict, sess, mode='auto'):
    '''
    {load weights to tf session}
    Input:
        weight_dict[dict]:
        mode[str]: 'auto', 'reuse', 'new'
    '''
    exist_varialbes = {}
    assign_ops = []
    for v in tf.all_variables():
        exist_varialbes[v.name] = v

    for k, v in weight_dict.items():
        if k in exist_varialbes.keys():
            if mode == 'new':
                raise KeyError('Variable already exist')
            tfv = exist_varialbes[k]
        else:
            if mode == 'reuse':
                raise KeyError('Variable Not Found')
            tfv = tf.get_variable(k.split(':')[0], shape=v.shape)
        op_ = tf.assign(tfv, v)
        assign_ops.append(op_)

    _ = sess.run(assign_ops)
