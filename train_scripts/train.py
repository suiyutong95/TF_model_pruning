import sys

sys.path.append('../.')
import os
import tensorflow as tf
from modeling.VHA_pruning import VHA_pruning_model
from model_zoo.losses import explogTVSK_loss_v3_binary as exp_v3b
from model_zoo.losses import explogTVSK_loss_v2 as exp_v2


class model(VHA_pruning_model):
    def __init__(self, sess):
        self.sess = sess
        self.seg_ch = 16

        self.segnet = 'dev'
        self.loss_func_main = lambda x, y: exp_v2(x, y, weight=1, alpha=.1, beta=.9, w_c=.8,w_d=.2)
        self.loss_func_hrt = lambda x, y: exp_v2(x, y, weight=1, alpha=.5, beta=.5, w_c=.2,w_d=.8)
        self.loss_func_vsl = lambda x, y: exp_v2(x, y, weight=1, alpha=.5, beta=.5, w_c=.2,w_d=.8)
        self.sliming_loss = True

        self.lr = 5e-4
        self.batch_size = 1
        self.depth = 144
        self.img_size = 192

        self.epoch = 500
        self.iters = 1000

        self.SAVE_DEBUGIMGS = False

        self.WEIGHT_DECAY = False
        self.DECAY_TYPE = 'WARMUP_PCD'
        self.EMA = True

        self.CLIP_GRADIENTS = True
        self.CLIP_GRADIENTS_THRESH = 5
        self.CLIP_GRADIENTS_MODE = 'LOCAL'

        self.prob_thresh = .8

        self.model_dir = '../../models_pruning/DEBUG'
        self.log_dir = os.path.join(self.model_dir, 'logs')
        self.ckpt_dir = os.path.join(self.model_dir, 'ckpts')

        self.pretrain_seg_epoch = 15
        self.reload_path = ''
        self.restore_seg_net = False

        # # pruning
        # self.pruning = False
        # self.pruning_radio = .5

        self.train_tfr_list = [
            ['../../data_mbridge/1st_run_train_160_256_pos.tfrecord', ],
        ]
        self.do_auglist = [True, ]
        self.test_tfr_list = [
            ['../../data_mbridge/1st_run_val_160_256_pos.tfrecord', ],
        ]
        self.do_auglist = [False, True, False, False, ]
        self.initial()

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    ''' 1st stage(warm-upstage) '''
    with tf.Session(config=config) as sess:
        nn = model(sess)
        nn.epoch = 30
        nn.loss_func_main = lambda x, y: exp_v2(x, y, weight=1, alpha=.1, beta=.9, w_c=.8,w_d=.2)
        nn.build()
        nn.train()

    ''' 2nd stage '''
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        nn = model(sess)
        nn.build()
        nn.train()
