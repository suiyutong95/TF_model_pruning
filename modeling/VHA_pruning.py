import os
import pickle, shelve
import copy

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .Segment_base import Segment_Base
from .NN_baseline import NN_baseline

from model_zoo.losses import explogTVSK_loss_v3_binary as exp_v3b
from model_zoo.net_frameworks.pruning import segnet_VHA_light
from model_zoo.net_frameworks.pruning.VHA_Unet_light_pruning_dev import segnet_VHA_light_dev

from pruning_utils.utils import P_node, save_graph, load_graph
from pruning_utils.rebuild_ops import rebuild_tf_graph


class VHA_pruning_model(Segment_Base):

    def __init__(self, sess):
        self.sess = sess
        self.seg_ch = 16

        self.segnet = 'default'
        self.loss_func_main = lambda x, y: exp_v3b(
            x, y, weight=100, alpha=.1, beta=.9, w_c=.2*240, label_smooth=.99, topk=3)
        self.loss_func_hrt = lambda x, y: exp_v3b(
            x, y, weight=100, alpha=.1, beta=.9, w_c=.2*240, label_smooth=.99, topk=3)
        self.loss_func_vsl = lambda x, y: exp_v3b(
            x, y, weight=100, alpha=.1, beta=.9, w_c=.2*240, label_smooth=.99, topk=3)
        self.sliming_loss = True

        self.lr = 1e-4
        self.batch_size = 2
        self.depth = 32
        self.img_size = 192

        self.epoch = 500
        self.iters = 1000

        self.SAVE_DEBUGIMGS = False

        self.WEIGHT_DECAY = True
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
            ['../../data_plaque_v2/rbk_small_patch/lst_run_train_40_120_pos.tfrecord', ],
            ['../../data_plaque_v2/rbk_small_patch/lst_run_train_40_120_pos.tfrecord', ],
            ['../../data_plaque_v2/rbk_small_patch/lst_run_train_40_330_pos.tfrecord', ],
            ['../../data_plaque_v2/rbk_small_patch/lst_run_train_40_330_nag.tfrecord', ],
        ]
        self.do_auglist = [False, True, False, False, ]

        self.test_tfr_list = [
            ['../../data_plaque_v2/rbk_small_patch/lst_run_val_40_120_pos.tfrecord', ],
        ]
        self.do_auglist = [False, True, False, False, ]

        self.initial()

    def initial(self):
        super().initial()
        self.SAMPLE_KEYS = ['image', 'label', 'image_id', 'vessel', 'heart']
        self.graph_dir = os.path.join(self.model_dir, 'graph')
        os.makedirs(self.graph_dir, exist_ok=True)

    def seg_net(self, x, is_training=False):
        if self.segnet == 'default':
            return segnet_VHA_light(x, base_channel=self.seg_ch, is_training=is_training,
                                    reuse=tf.AUTO_REUSE, upsample_type='resize')
        elif self.segnet == 'dev':
            return segnet_VHA_light_dev(x, base_channel=self.seg_ch, is_training=is_training,
                                    reuse=tf.AUTO_REUSE, upsample_type='resize')
        else:
            raise NameError('Un-known segnet name')

    def _save_graph(self, graph):
        save_graph(graph,os.path.join(self.graph_dir, 'graph.pkl'))

    def _load_graph(self):
        graph = load_graph(os.path.join(self.graph_dir, 'graph.pkl'))
        return graph

    def _build_net(self, batched_data, is_training=False):
        '''return x_out[dict[key:tf.tensor]],'''
        x = tf.expand_dims(batched_data['image'], -1)
        x.set_shape([self.sub_batch_size, self.depth, self.img_size, self.img_size, 1])
        # re-build
        if 0:
            graph = self._load_graph()
            graph_rb = rebuild_tf_graph(x, graph)
            return {
                'probs': graph_rb.output_nodes['OP'],
                'vsl_probs': graph_rb.output_nodes['V_OP'],
                'hrt_probs': graph_rb.output_nodes['H_OP'],
            }
        # first-build
        else:
            x = P_node(x, y=None, is_head=True)
            with tf.variable_scope('SEG_NET', reuse=tf.AUTO_REUSE):
                OP, V_OP, H_OP, _ = self.seg_net(x, is_training=is_training)
            #
            self._save_graph(OP.graph, )

            return {
                'probs': OP.tensor_out,
                'vsl_probs': V_OP.tensor_out,
                'hrt_probs': H_OP.tensor_out,
            }

        # if is_training:
        #     x = P_node(x, y=None, is_head=True)
        #     with tf.variable_scope('SEG_NET', reuse=tf.AUTO_REUSE):
        #         OP, V_OP, H_OP, _ = self.seg_net(x, is_training=is_training)
        #     #
        #     self._save_graph(OP.graph,)
        #
        #     return {
        #         'probs': OP.tensor_out,
        #         'vsl_probs': V_OP.tensor_out,
        #         'hrt_probs': H_OP.tensor_out,
        #     }
        # else:   # for test, we do not need prune graph
        #     with tf.variable_scope('SEG_NET', reuse=tf.AUTO_REUSE):
        #         OP, V_OP, H_OP, _ = self.seg_net(x, is_training=is_training)
        #
        #     return {
        #         'probs': OP,
        #         'vsl_probs': V_OP,
        #         'hrt_probs': H_OP,
        #     }


    @NN_baseline.LossWrapper
    def _get_loss(self, outputs, batched_data):
        '''
        outputs[dict]->
        batched_data[dict]->

        return loss_dict[key:loss] must have key 'total_loss'
        '''
        label = batched_data['label']
        vessel = batched_data['vessel']
        heart = batched_data['heart']

        pred = outputs['probs']
        vsl_pred = outputs['vsl_probs']
        hrt_pred = outputs['hrt_probs']

        # seg loss
        seg_loss = self.loss_func_main(label, pred)
        vsl_loss = self.loss_func_main(vessel, vsl_pred)
        hrt_loss = self.loss_func_main(heart, hrt_pred)

        # sliming_loss
        sliming_loss = 0
        if self.sliming_loss:
            for v in self._get_trainable_vars():
                if 'gamma' in v.name:
                    sliming_loss += tf.nn.l2_loss(v)

        total_loss = seg_loss+.1*vsl_loss+.1*hrt_loss+1e-7*sliming_loss

        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'vsl_loss': vsl_loss,
            'hrt_loss': hrt_loss,
            'sliming_loss': sliming_loss,
        }

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        nn = VHA_pruning_model(sess)
        nn.build()
        nn.train()
