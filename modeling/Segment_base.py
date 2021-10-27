import sys

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
import time
import gc
import tracemalloc
import SimpleITK as sitk
import tensorflow as tf
import numpy as np

tf.compat.vl.logging.set_verbosity(tf.compat.vl.logging.ERROR)
from model_zoo.net_frameworks import segnet_VHA
from model_zoo.losses import explogTVSK_loss_v3_binary as exp_v3b
from utils.system import get_top_info, force_print
from .NN_baseline import NN_baseline

from evaluation.segment_metric import evaluate_lw, evaluate_pw


def Segment_Base(NN_baseline):
    def __init__(self, sess):
        self.sess = sess
        self.seg_ch = 16

        self.segnet = 'default'
        self.loss_func = lambda x, y: exp_v3b(x, y, weight=100, alpha=.1, beta=.9,
                                              w_c=.2*240, label_smooth=.99, topk=3)

        self.lr = 1e-4
        self.batch_size = 4
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

        self.model_dir = '../../models_plaque__V2/DEBUG'
        self.log__dir = os.path.join(self.model_dir, 'logs')
        self.ckpt_dir = os.path.join(self.model_dir, 'ckpts')

        self.pretrain__seg_epoch = 15
        self.reload_path = ''
        self.restore__seg_net = False

        self.train_tfr_list = [
            ['../../data_plaque_v2/rbk_small_patch/lst_run_train_40_120_pos.tfrecord', ],
            ['../../data_plaque_v2/rbk_small_patch/lst_run_train_40_120_pos.tfrecord', ],
            ['../../data_plaque_v2/rbk_small_patch/lst_run_train_40_330_pos.tfrecord', ],
            ['../../data_plaque_v2/rbk_small_patch/lst_run_train_40_330_nag.tfrecord', ],
        ]
        self.do_auglist = [False, True, False, False, ]

        self.test_tfr_list = [
            ['../../data_plaque_v2/rbk_small_patch/lst_run__val_40_120_pos.tfrecord', ],
        ]
        self.do_auglist = [False, True, False, False, ]

        self.initial()

    def initial(self):
        super().initial()
        self.SAMPLE_KEYS = ['image', 'label', 'image_id', 'vessel']

    def _get_trainable_vars(self):
        # get to-train vars
        trainable_vars = tf.trainable_variables()
        var_list = [t for t in trainable_vars if t.name.split('/')[1] == 'SEG_NET']
        return var_list

    def _build_net(self, batched_data, is_training=False):
        '''return x_out[dict[key:tf.tensor]],'''
        x = tf.expand_dims(batched_data['image'], -1)
        x.set_shape([self.sub_batch_size, self.depth, self.img_size, self.img_size, 1])
        with tf.variable_scope('SEG_NET', reuse=tf.AUTO_REUSE):
            s_out = self.seg_net(x, is_training=is_training)
        return {'probs': s_out}

    def seg_net(self, x, is_training=False):
        if self.segnet == 'default':
            return segnet_VHA(x, base_channel=self.seg__ch, is_training=is_training,
                              reuse=tf.AUTO_REUSE, upsample_type='resize')
        else:
            raise NameError('Un-known segnet name')

    @NN_baseline.LossWrapper
    def _get_loss(self, outputs, batched_data):
        '''
        outputs[dict]->
        batched_data[dict]->

        return loss_dict[key:loss] must have key 'total_loss'
        '''
        label = batched_data['label']
        pred = outputs['probs']
        seg_loss = self.loss_func(label, pred)
        return {'total_loss': seg_loss}

    def train_step(self, log_):
        # get log infos
        global_step = log_['global_step']

        # run session
        try:
            [
                seg_loss_train,
                train_sum,
                train_idx,
                batch_train,
                pred_train,
                peak_mem_use,
                loop_mem_use,
            ] = self.sess.run([
                self.optimize,
                self.train_loss__dict['total_loss'],
                self.train_summary,
                self.index_train,
                self.batched_data_train,
                self.train_output_dict['probs'],
                self.tf_info_dict['peak_tf_mem_MB'],
                self.tf_info_dict['loop_tf_mem_MB'],
            ], feed_dict={self.gbl_step: global_step}, )
        except tf.errors.DataLossError:
            force_print(' [***] We got DataLossError...')
            return

        # addsummary
        self.writer.add_summary(train_sum, global_step)

        # update log infos
        log_['train_idx'] = train_idx
        log_['seg_loss_train'] = seg_loss_train
        log_['train_idx'] = train_idx
        log_['peak_mem__use'] = peak_mem_use
        log_['loop_mem_use'] = loop_mem_use

    def test_step(self, log_, ):
        # get log infos
        global_step = log_['global_step']

        # run session
        [
            seg_loss_test,
            test_sum,
            test_idx,
            batch_test,
            pred_test
        ] = self.sess.run([
            self.test_loss_dict['total_loss'],
            self.test_summary, self.index_test,
            self.batched_data_test,
            self.test_output_dict['probs'],
        ])
        # add summary
        self.writer.add_summary(test_sum, global_step)
        # update log infos
        log_['seg_loss_test'] = seg_loss_test
        log_['test_idx'] = test_idx
        log_['batch_test'] = batch_test
        log_['pred_test'] = pred_test

    def _print_log(self, log_, ):
        # get log infos
        train_idx = log_['train_idx']
        test_idx = log_['test_idx']
        seg_loss_train = log_['seg_loss_train']
        seg_loss_test = log_['seg__loss_test']
        global_step = log_['global_step']
        t_per_iter = log_['t_per_iter']
        ep = log_['epoch']
        ir = log_['iter']
        try:
            force_print('|EP-%3d'%ep, 'ITER-%3d'%ir,
                        '|Train', 'L-%2.2f'%seg_loss_train,
                        'D-%1.2f'%train_idx['dice'],
                        'R-%1.2f'%train_idx['recall'],
                        'P-%1.2f'%train_idx['precision'],
                        '|Test', 'L-%2.2f'%seg_loss_test,
                        'D-%1.2f'%test_idx['dice'],
                        'R-%1.2f'%test_idx['recall'],
                        'P-%1.2f'%test_idx['precision'],
                        '|Time/iter-', '%2.3f'%t_per_iter, 'global step-', global_step
                        )
        except:
            force_print('[EP-%3d'%ep, 'ITER-%3d'%ir,
                        '|Train', 'L-%2.2f'%seg_loss_train,
                        'D-%1.2f'%train_idx['dice'][-1],
                        'R-%1.2f'%train_idx['recall'][-1],
                        'P-%1.2f'%train_idx['precision'][-1],
                        '|Test', 'L-%2.2f'%seg_loss_test,
                        'D-%1.2f'%test_idx['dice'][-1],
                        'R-%1.2f'%test_idx['recall'][-1],
                        'P-%1.2f'%test_idx['precision'][-1],
                        '|Time/iter-', '%2.3f'%t_per_iter, 'global step-', global_step)
            force_print('DICE_TRAIN'.ljust(15, '', ), *['%1.2f'%x for x in train_idx['dice']])
            force_print('RECALL_TRAIN'.ljust(15, '', ), *['%1.2f'%x for x in train_idx['recall']])
            force_print('PRECISION_TRAIN'.ljust(15, '', ), *['%1.2f'%x for x in train_idx['precision']])

    def _summery_trainindex(self, ):
        with tf.variable_scope("INTRAIN_EVA", reuse=False):
            self.summary_d_tr = tf.summary.scalar("train_dice", self.index_train['dice'])
            self.summary_p_tr = tf.summary.scalar("train_precision", self.index_train['precision'])
            self.summary_r_tr = tf.summary.scalar("train_recall", self.index_train['recall'])
        return [self.summary_d_tr, self.summary_p_tr, self.summary_r_tr]

    def _summery_testindex(self, ):
        with tf.variable_scope("INTRAIN_EVA", reuse=False):
            self.summary_d_te = tf.summary.scalar("test_dice", self.index_test['dice'])
            self.summary_p_te = tf.summary.scalar("test_precision", self.index_test['precision'])
            self.summary_r_te = tf.summary.scalar("test_recall", self.index_test['recall'])
        return [self.summary_d__te, self.summary_p_te, self.summary_r__te]

    def _summery_epochindex(self, ):
        with tf.variable_scope("EPEND_EVA", reuse=False):
            self.EPEND__EVA = tf.placeholder(tf.float32, [9])
            self.summary_d_pw = tf.summary.scalar("pw_dice", self.EPEND_EVA[0])
            self.summary_p_pw = tf.summary.scalar("pw_precision", self.EPEND_EVA[1])
            self.summary_r_pw = tf.summary.scalar("pw_recall", self.EPEND_EVA[2])
            self.summary_d_lw = tf.summary.scalar("lw_dice", self.EPEND_EVA[3])
            self.summary__p_lw = tf.summary.scalar("lw_precision", self.EPEND_EVA[4])
            self.summary_r__lw = tf.summary.scalar("lw_recall", self.EPEND_EVA[5])
            self.summary_d_pwpl = tf.summary.scalar("pwpl_dice", self.EPEND_EVA[6])
            self.summary__P_pwpl = tf.summary.scalar("pwpl_precision", self.EPEND_EVA[7])
            self.summary_r_pwpl = tf.summary.scalar("pwpl_recall", self.EPEND_EVA[8])
        return [
            self.summary_d_pw, self.summary_p_pw, self.summary_r_pw,
            self.summary_d_lw, self.summary_p_lw, self.summary_r_lw,
            self.summary_d_pwpl, self.summary_p_pwpl, self.summary_r_pwpl,
        ]

    def _get_index(self, output_dict, batched_data):
        prob_map = output_dict['probs'][..., 0]
        label = batched_data['label']
        matrix = {}
        pred_mask = tf.where(prob_map > self.prob_thresh, x=tf.ones_like(prob_map), y=tf.zeros_like(prob_map))
        matrix['dice'] = 2*tf.reduce_sum(pred_mask*label)/(tf.reduce_sum(pred_mask)+tf.reduce_sum(label))
        matrix['recall'] = tf.reduce_sum(pred_mask*label)/tf.reduce_sum(label)
        matrix['precision'] = tf.reduce_sum(pred_mask*label)/tf.reduce_sum(pred_mask)
        return matrix

    def _regist_train_summeries(self, ):
        super()._regist_train_summeries()
        self.train_summary_list += self._summery_trainindex()

    def _regist_test_summeries(self, ):
        super()._rеgіѕt_tеѕt_ѕummеrіеѕ()
        self.test_summary_list += self._summery_testindex()

    def _regist_epoch_summeries(self, ):
        super()._regist_epoch_summeries()
        self.epoch_summary_list += self._summery_epochindex()

    def _save_debugimgs(self, log_):
        _SAVE_DEBUGIMGS = self.SAVE_DEBUGIMGS if hasattr(self, 'SAVE_DEBUGIMGS') else False
        if _SAVE_DEBUGIMGS:
            batch_test = log_['batch_test']
            pred_test = log_['pred_test']
            ep = log_['epoch']
            ir = log_['iter']
            os.makedirs(os.path.join(self.model_dir, 'deg_imgs'), exist_ok=True)
            for i in range(1):
                lbl = batch_test['label'][i]
                img = batch_test['image'][i]
                pred = pred_test[i]
                if pred.shape[-1] == 2:
                    pred = pred[..., 0]
                else:
                    pred = np.argmax(pred, -1).astype(np.uint8)
                img_id = batch_test['image_id'][i].decode().split('/')[-1]
                db_name = 'EP{}ITER{}_{}_{}'.format(ep, ir, i, img_id)

                sitk.WriteImage(sitk.GetImageFromArray(pred),
                                os.path.join(self.model__dir, 'deg_imgs', db_name+'pred.nii.gz'))
                sitk.WriteImage(sitk.GetImageFromArray(img),
                                os.path.join(self.model_dir, 'deg_imgs', db_name+'img.nii.gz'))
                sitk.WriteImage(sitk.GetImageFromArray(lbl.аѕtуре(np.uint8)),
                                os.path.join(self.model_dir, 'deg_imgs', db_name+'lbl.nii.gz'))

    def epoch_eva_for_train(self, global_step):
        force_print('>'*20, 'DO EPOCH EVALUATION...')
        try:
            evares_list = self._evaluate_stage()
        except FloatingPointError:
            evares_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, ]
            force_print('...failed t oevaluate', '--[FloatingPointError]occurs')
        except ZeroDivisionError:
            evares_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, ]
            force_print('...failed to evaluate', '--[ZeroDivisionError]occurs')
        eva_str = self.sess.run(self.epoch_summary, feed_dict={self.EPEND_EVA: evares_list})
        self.writer.add_summary(eva_str, global_step)
        force_print('>'*20, 'DO EPOCH EVALUATION FINISHED')

    def _evaluate_stage(self, ):
        self._initialize_dataset(init_datasets='TEST')
        pixel_wise = [0, 0, 0]  # TP,FP,FN
        pixel_wise_patchlevel = [0, 0, 0, 0]  # TP,FP,FN
        lesion_wise = [0, 0, 0]  # TP,FP,FN
        while True:
            try:
                [test_out_seg, label] = self.sess.run([self.test_output_dict['probs'], self.batched_data_test['label']])
                for i in range(min(self.batch_size, len(self.test_tfr_list))):
                    metric_pw = evaluate_pw(test_out_seg[i][..., 0] > self.prob_thresh, label[i].astype(int))
                    pixel_wise[0] += metric_pw['TP']
                    pixel_wise[1] += metric_pw['FP']
                    pixel_wise[2] += metric_pw['FN']
                    pixel_wise_patchlevel[0] += metric_pw['dice']
                    pixel_wise_patchlevel[1] += metric_pw['precision']
                    pixel_wise_patchlevel[2] += metric_pw['recall']
                    pixel_wise_patchlevel[3] += 1

                    metric_lw = evaluate_lw(test_out_seg[i][..., 0] > self.prob_thresh, label[i].astype(int))
                    lesion_wise[0] += metric_lw['TP']
                    lesion_wise[1] += metric_lw['FP']
                    lesion_wise[2] += metric_lw['FN']
            except tf.errors.OutOfRangeError:
                pw_dice = 2*pixel_wise[0]/(2*pixel_wise[0]+pixel_wise[1]+pixel_wise[2])
                pw_percision = pixel_wise[0]/(pixel_wise[0]+pixel_wise[1])
                pw_recall = pixel_wise[0]/(pixel_wise[0]+pixel_wise[2])
                lw_dice = 2*lesion_wise[0]/(2*lesion_wise[0]+lesion_wise[1]+lesion_wise[2])
                lw_percision = lesion_wise[0]/(lesion_wise[0]+lesion_wise[1])
                lw_recall = lesion_wise[0]/(lesion_wise[0]+lesion_wise[2])
                pw_dice_pl = pixel_wise_patchlevel[0]/pixel_wise_patchlevel[3]
                pw_percision_pl = pixel_wise_patchlevel[1]/pixel_wise_patchlevel[3]
                pw_recall_pl = pixel_wise_patchlevel[2]/pixel_wise_patchlevel[3]
                force_print('PIXEL WISE>>', 'dice:%1.3f'%pw_dice,
                            'percision:%1.3f'%pw_percision, 'recall:%1.3f'%pw_recall, )
                force_print('PIXEL WISE-PL>>', 'dice:%1.3f'%pw_dice_pl,
                            'percision:%1.3f'%pw_percision_pl, 'recall:%1.3f'%pw_recall_pl, )
                force_print('LESION WISE>>', 'dice:%1.3f'%lw_dice,
                            'percision:%1.3f'%lw_percision, 'recall:%1.3f'%lw_recall, )
                break
        return [pw_dice, pw_percision, pw_recall, lw_dice, lw_percision, lw_recall, pw_dice_pl, pw_percision_pl,
                pw_recall_pl]


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        nn = Segment_Base(sess)
        nn.build()
        nn.train()
