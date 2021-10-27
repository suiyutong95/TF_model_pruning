import tensorflow as tf


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>augs>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def random_flip_l_r_3(inlist):
    z = tf.constant(0)
    r = tf.random_uniform([], 0, 2, tf.int32)
    outlist = []
    for x in inlist:
        x_ = tf.cond(z < r, lambda: x, lambda: x[..., ::-1])
        outlist.append(x_)
    return outlist


def _3d_rotate(x, angle, interpolation):
    '''x->[z,y,x]'''
    x = tf.transpose(x, (1, 2, 0))
    x = tf.contrib.image.rotate(x, angle, interpolation=interpolation)
    x = tf.transpose(x, (2, 0, 1))
    return x


def random_rotate_l_r_3(inlist):
    '''[z,y,x]'''
    angle_lim = 1
    angle = tf.random_normal([], mean=0, stddev=(angle_lim/3)*3.14/180, dtype=tf.float32)
    outlist = []
    for i, x in enumerate(inlist):
        if i == 0:
            x_ = _3d_rotate(x, angle, interpolation='BILINEAR')
        else:
            x_ = _3d_rotate(x, angle, interpolation='NEAREST')
        outlist.append(x_)
    return outlist


def _zoom_affine(x, size, interpolation='BILINEAR'):
    '''x->[x,y,c]'''
    image_shape = tf.cast(tf.shape(x), tf.float32)
    zsize = tf.cast(size, tf.float32)
    h_frac = image_shape[0]/zsize[0]
    w_frac = image_shape[1]/zsize[1]
    hd = 0.5*h_frac*(zsize[0]-image_shape[0])
    wd = 0.5*w_frac*(zsize[1]-image_shape[1])
    zoom_tr = tf.convert_to_tensor([h_frac, 0, hd, 0, w_frac, wd, 0, 0])
    zoom_tr = tf.expand_dims(zoom_tr, axis=0)
    return tf.contrib.image.transform(x, zoom_tr, interpolation=interpolation)


def _zoom_3Dxy(x, zoom_pxl, method='BILINEAR'):
    '''Input[zyx]:'''
    shape = tf.cast(tf.shape(x), tf.float32)
    x = tf.transpose(x, (1, 2, 0))
    x = _zoom_affine(x, [shape[1]+zoom_pxl, shape[2]+zoom_pxl], interpolation=method)
    x = tf.transpose(x, (2, 0, 1))
    return x


def random_zoom_xy(inlist):
    pxl_lim = 12
    zoom_pxl = tf.random_normal([], mean=0, stddev=pxl_lim//3, dtype=tf.float32)
    # zoom_pxl = tf.cast(zoom_pxl, tf.int32)
    outlist = []
    for i, x in enumerate(inlist):
        if i == 0:
            x_ = _zoom_3Dxy(x, zoom_pxl, method='BILINEAR')
        else:
            x_ = _zoom_3Dxy(x, zoom_pxl, method='NEAREST')
        outlist.append(x_)

    return outlist


#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>crops>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def pickrandom2(array_list, img_size, depth=8, size=384):
    '''
    [PICK RANDOM SLICES, *image and label at same time]
    Input:
        array_list[list[tf.Tensor]]->[list[depth,x-axis,y-axis]]: list of arrays to-be cropped
        img_size[list]->Sizeof3-dimage
        depth[int]->targetdepth
        size[int]->targetx,yaxis
    Output:
        array_list_cropped->
    '''
    condi = tf.Assert(tf.greater_equal(img_size[0], depth), ['Image depth smaller than target depth'])
    array_list_cropped = []
    with tf.control_dependencies([condi]):
        st_idx_d = tf.random_uniform(
            [1], maxval=img_size[0]-depth+1, dtype=tf.int32)[0]
        st_idx_x = tf.random_uniform(
            [1], maxval=img_size[1]-size+1, dtype=tf.int32)[0]
        st_idx_y = tf.random_uniform(
            [1], maxval=img_size[1]-size+1, dtype=tf.int32)[0]
    for x in array_list:
        array_list_cropped.append(x[
                                  st_idx_d:st_idx_d+depth,
                                  st_idx_x:st_idx_x+size,
                                  st_idx_y:st_idx_y+size
                                  ])
    shift = tf.stack([st_idx_d, st_idx_x, st_idx_y, st_idx_d, st_idx_x, st_idx_y])
    return array_list_cropped


def pickmid_ingraph(image, img_size, depth=8, size=384):
    '''
    三维图像对depth修改
    仅用于TF图内计算
    [PICKMIDSLICES]
    Input:
        img[np.array/tf.Tensor] -> [depth,x-axis,y-axis]
        img_size[list] -> Size of 3-dimage
        depth[int] -> target depth
    Output:
        img_out -> [target_depth,x-axis,y-axis]
    '''
    condi = tf.Assert(tf.greater_equal(img_size[0], depth), ['Image depth smaller than target depth'])
    with tf.control_dependencies([condi]):
        start_idx = (img_size[0]-depth)//2
        start_idx2 = (img_size[1]-size)//2
        img_out = image[
                  start_idx:start_idx+depth,
                  start_idx2:start_idx2+size,
                  start_idx2:start_idx2+size
                  ]
        return img_out
