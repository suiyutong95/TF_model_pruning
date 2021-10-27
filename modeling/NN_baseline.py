import sys
sys.path.append('../.')
sys.path.append('../..')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE']='1'
import time
import gc
import tracemalloc
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .Base import GPUParallelBase
from .dataloader.iterator_V3 import make_batch_iterator_V3
from .nn_utils import gradient_clip
from .nn_utils import average_gradients,assign_to_device,get_available_gpus

class NN_baseline(GPUParallelBase):
    def _build_optimizer(self):
        _ALLOW_MIXED_PRECISION = self.ALLOW_MIXED_PRECISION if hasattr(self, 'ALLOW_MIXED_PRECISION') else True
        _OPTIMIZER = self.OPTIMIZER if hasattr(self, 'OPTIMIZER') else 'Adam'
        _DECAY_TYPE = self.DECAY_TYPE if hasattr(self, 'DECAY_TYPE') else 'periodic_cos_decay'
        self.gbl_step = tf.placeholder(tf.int32, [], name='global_step')
        if _DECAY_TYPE == 'cos_decay':
            self.lr_decay = tf.train.cosine_decay(self.lr, self.gbl_step, 350*self.iters, alpha=0.2)
        elif _DECAY_TYPE == 'periodic_cos_decay':
            N = 200000
            GSP = tf.cond(tf.mod(self.gbl_step, 2*N) > N,
                          lambda: 2*N-tf.mod(self.gbl_step, 2*N),
                          lambda: tf.mod(self.gbl_step, 2*N)
            )
            self.lr_decay = tf.train.cosine_decay(self.lr, GSP, N, alpha=0.2)
        elif _DECAY_TYPE == 'WARMUP_PCD':
            # periodic cos decay with warmup
            N = 200000
            warmup_step = 1000
            GSP = tf.cond(tf.mod(self.gbl_step, 2*N) > N,
                          lambda: 2*N-tf.mod(self.gbl_step, 2*N),
                          lambda: tf.mod(self.gbl_step, 2*N)
            )
            self.lr_decay = tf.cond(
                self.gbl_step < warmup_step,
                lambda: tf.train.cosine_decay(self.lr, warmup_step-self.gbl_step, warmup_step, alpha=0.2),
                lambda: tf.train.cosine_decay(self.lr, GSP, N, alpha=0.2),
            )
        else:
            raise NameError('UndefinedLRDECAYTYPEname')
        if _OPTIMIZER == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_decay)
        elif _OPTIMIZER == 'Momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.lr_decay, .99)
        else:
            raise NameError('UndefinedOPTIMIZERname')

        if _ALLOW_MIXED_PRECISION:
            self.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.optimizer)

    def _build_index(self):
        with tf.variable_scope("index", reuse=tf.AUTO_REUSE), tf.device(self.controller):
            self.index_train = self._get_index(self.train_output_dict, self.batched_data_train)
            self.index_test = self._get_index(self.test_output_dict, self.batched_data_test)


    def LossWrapper(func):
        '''add weight decay'''
        def _wrapper(self, *args, **kwargs):
            _WEIGHT_DECAY = self.WEIGHT_DECAY if hasattr(self, 'WEIGHT_DECAY') else False
            if _WEIGHT_DECAY:
                l2_norm_ = []
                for v in self._get_trainable_vars():
                    l2_norm_.append(tf.nn.l2_loss(v))
                l2_norm = tf.add_n(l2_norm_)
                loss_dict = func(self, *args, **kwargs)
                loss_dict['total_loss'] += l2_norm
                loss_dict['weight_decay'] = l2_norm
                return loss_dict
            else:
                return func(self, *args, **kwargs)
        return _wrapper

    def _wrapper_gradients(self, grads):
        '''clip gradients'''
        _CLIP_GRADIENTS = self.CLIP_GRADIENTS if hasattr(self, 'CLIP_GRADIENTS') else False
        _CLIP_GRADIENTS_THRESH = self.CLIP_GRADIENTS_THRESH if hasattr(self, 'CLIP_GRADIENTS_THRESH')  else 5
        _CLIP_GRADIENTS_MODE = self.CLIP_GRADIENTS_MODE if hasattr(self, 'CLIP_GRADIENTS_MODE') else 'LOCAL'
        if _CLIP_GRADIENTS:
            return gradient_clip(grads,clip_norm=_CLIP_GRADIENTS_THRESH,mode=_CLIP_GRADIENTS_MODE)
        else:
            return grads

    def _wrapper_iterator(self, *args, **kwargs):

        kwargs['slice_depth'] = self.depth
        kwargs['slice_size'] = self.img_size
        return make_batch_iterator_V3(*args, **kwargs)

    def _reconstract_batch_output(self, ):
        self.train_output_dict = {}
        self.test_output_dict = {}
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        with tf.name_scope("rec_op"), tf.device(self.controller):
            for key in self.train_output_dict_:
                self.train_output_dict[key] = tf.concat(self.train_output_dict_[key], axis=0)
            for key in self.test_output_dict_:
                self.test_output_dict[key] = tf.concat(self.test_output_dict_[key], axis=0)
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE), tf.device(self.controller):
            for key in self.train_loss_dict_:
                self.train_loss_dict[key] = tf.reduce_mean(self.train_loss_dict_[key],)
            for key in self.test_loss_dict_:
                self.test_loss_dict[key] = tf.reduce_mean(self.test_loss_dict_[key],)

    def _summery_LR(self, ):
        with tf.variable_scope("LR", reuse=False):
            return [tf.summary.scalar("learning rate", self.lr_decay)]


    def _summery_trainloss(self, ):

        _summery_loss_keys = self.SUMMERY_LOSS_KEYS if hasattr(self, 'SUMMERY_LOSS_KEYS') \
                                                   else list(self.train_loss_dict.keys())
        with tf.variable_scope("TrainLoss", reuse=False):
            summery_list = []
            for key in _summery_loss_keys:
                summery_list.append(tf.summary.scalar(key, self.train_loss_dict[key]))
        return summery_list

    def _summery_testloss(self, ):
        _summery_loss_keys = self.SUMMERY_LOSS_KEYS if hasattr(self, 'SUMMERY_LOSS_KEYS')  \
                                                   else list(self.test_loss_dict.keys())
        with tf.variable_scope("TestLoss", reuse=False):
            summery_list = []
            for key in _summery_loss_keys:
                summery_list.append(tf.summary.scalar(key, self.test_loss_dict[key]))
            return summery_list

    def _regist_train_summeries(self,):
        super()._regist_train_summeries()
        self.train_summary_list += self._summery_LR()
        self.train_summary_list += self._summery_trainloss()

    def _regist_test_summeries(self,):
        super()._regist_test_summeries()
        self.test_summary_list += self._summery_testloss()