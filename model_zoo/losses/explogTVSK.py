import tensorflow as tf


def explogTVSK_loss_v3_binary(
        gt, pred, weight=120,
        w_d=.8, w_c=.2, gm_d=.3, gm_c=.3, alpha=.1, beta=.9,
        eps=1., topk=False, label_smooth=False
):
    '''
    {}
    Params:
        topk[False or int]:
        label_smooth[False or float]:
    Returns:
        loss[tf.tensor]:
    '''
    batch_size = tf.shape(gt)[0]
    gt = tf.cast(gt, dtype=tf.float32)
    p0 = tf.reshape(pred, (batch_size, -1, 2))[..., 0]
    g0 = tf.reshape(gt, (batch_size, -1, 1))[..., 0]

    if label_smooth:
        g0 = g0*label_smooth

    gl = tf.ones_like(g0)-g0
    p1 = tf.ones_like(p0)-p0
    dice = (tf.reduce_sum(p0*g0, axis=-1)+eps)/(tf.reduce_sum(p0*g0, axis=-1)+
                                                alpha*tf.reduce_sum(p0*gl, axis=-1)+
                                                beta*tf.reduce_sum(p1*g0, axis=-1)+eps)
    prob = g0*p0+gl*p1
    weight_mask = g0*weight+gl*1
    prob_smooth = tf.clip_by_value(prob, 1e-15, 1.0-1e-7)

    LD = tf.pow(-tf.log(dice), gm_d)
    LC = tf.reduce_sum(tf.pow(-tf.log(prob_smooth), gm_c)*weight_mask, axis=-1)/ \
         tf.cast(tf.shape(prob_smooth)[1], dtype=tf.float32)
    explog = LD*w_d+LC*w_c
    if topk:
        topk_explog = tf.cond(topk < batch_size, lambda: tf.math.top_k(explog, topk).values, lambda: explog)
        return tf.reduce_mean(topk_explog)
    else:
        return tf.reduce_mean(explog)
