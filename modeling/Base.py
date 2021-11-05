import sys

sys.path.append('../.')
sys.path.append('../..')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
import time
import gc
import tracemalloc
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils.system import get_top_info, force_print
from .nn_utils import average_gradients, assign_to_device, get_available_gpus


class GPUParallelBase():
    def __init__(self, sess):
        self.sess = sess
        self.model_dir = './'

    def initial(self):

        self.log_dir = os.path.join(self.model_dir, 'logs')
        self.ckpt_dir = os.path.join(self.model_dir, 'ckpts')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.devices = get_available_gpus()
        self.n_gpus = len(self.devices)
        self.sub_batch_size = int(self.batch_size/self.n_gpus)

    def _build_net(self, batched_data, is_training=False):
        '''
        return x_out[dict[key:tf.tensor]],
        '''

        raise NotImplementedError

    def _build_optimizer(self):
        '''
        define self.optimizer
        '''
        raise NotImplementedError

    def _get_loss(self, outputs, batched_data):
        '''
        outputs[dict] ->
        batched_data[dict] ->
        return loss_dict[key:loss] -> must have key 'total_ loss "
        '''
        raise NotImplementedError

    def _get_index(self, output_dict, batched_data):
        '''
        calculate output index
        return matrix
        '''
        raise NotImplementedError

    @staticmethod
    def _get_trainable_vars():
        # get to-train vars
        trainable_vars = tf.trainable_variables()
        return trainable_vars

    def _wrapper_gradients(self, grads):
        '''
        wrap gradients
        '''
        return grads

    def _wrapper_iterator(self, *args, **kwargs):
        raise NotImplementedError

    def _build_dataloader(self):
        _SAMPLE_KEYS = self.SAMPLE_KEYS if hasattr(self, 'SAMPLE_KEYS') else []
        _do_auglist = self.do_auglist if hasattr(self, 'do_auglist') else [False]
        force_print('>'*20, 'ALLOCATING BATCH DATA ...')
        force_print('SAMPLE_KEYS ->', [x for x in _SAMPLE_KEYS])

        # train data loader
        self.train_iterator_list = []
        self.batched_data_train = {}
        _dict = {}
        for key in _SAMPLE_KEYS:
            _dict[key] = []
        force_print('TRAIN DATA --')
        for i in range(self.batch_size):
            force_print('-Batch', i, 'DATA -> ', self.train_tfr_list[i%len(self.train_tfr_list)])
            iterator = self._wrapper_iterator(
                self.train_tfr_list[i%len(self.train_tfr_list)],
                batch_size=1, is_train=True,
                aug_on=_do_auglist[i%len(_do_auglist)],
            )
            self.train_iterator_list.append(iterator)
            sample = iterator.get_next()
            for key in _SAMPLE_KEYS:
                _dict[key].append(sample[key])
        for key in _SAMPLE_KEYS:
            self.batched_data_train[key] = tf.concat(_dict[key], 0)

        # test data loader
        self.test_iterator_list = []
        self.batched_data_test = {}
        _dict = {}
        for key in _SAMPLE_KEYS:
            _dict[key] = []
        force_print('TEST DATA --')
        for i in range(self.batch_size):
            force_print('-Batch', i, 'DATA -> ', self.test_tfr_list[i%len(self.test_tfr_list)])
            iterator = self._wrapper_iterator(
                self.test_tfr_list[i%len(self.test_tfr_list)],
                batch_size=1, is_train=False,
                # aug_on=_do_auglist[i%len(_do_auglist)],
            )
            self.test_iterator_list.append(iterator)
            sample = iterator.get_next()
            for key in _SAMPLE_KEYS:
                _dict[key].append(sample[key])
        for key in _SAMPLE_KEYS:
            self.batched_data_test[key] = tf.concat(_dict[key], 0)

    def _initialize_dataset(self, init_datasets='ALL'):
        _no_initialize_train_loader = True
        if init_datasets == 'ALL':
            initializers = [iterator.initializer for iterator in self.train_iterator_list]+ \
                           [iterator.initializer for iterator in self.test_iterator_list]
        elif init_datasets == 'TEST':
            initializers = [iterator.initializer for iterator in self.test_iterator_list]
        elif init_datasets == 'TRAIN':
            initializers = [iterator.initializer for iterator in self.train_iterator_list]
        else:
            raise NameError("Un-defined to-be-initialize datasets' name")
        self.sess.run(initializers)
        force_print(init_datasets, 'Dataset Initialized ..')

    def _regist_train_summeries(self, ):
        self.train_summary_list = []

    def _regist_test_summeries(self, ):
        self.test_summary_list = []

    def _regist_epoch_summeries(self, ):
        self.epoch_summary_list = []

    def _build_summary_graph(self):
        # BuildSummery
        self._regist_train_summeries()
        self._regist_test_summeries()
        self._regist_epoch_summeries()
        # MergeSummery
        self.train_summary = tf.summary.merge(self.train_summary_list)
        self.test_summary = tf.summary.merge(self.test_summary_list)
        self.epoch_summary = tf.summary.merge(self.epoch_summary_list)

    def _build_devicelog_graph(self):
        with tf.variable_scope("DEVICE_LOG", reuse=False):
            self.device_log = tf.placeholder(tf.float32, [8])
            self.step_duration = tf.summary.scalar("step time", self.device_log[0])
            self.cpu_usage = tf.summary.scalar("cpu usage", self.device_log[1])
            self.mem_usage = tf.summary.scalar("memery usage", self.device_log[2])
            self.total_load = tf.summary.scalar("total load", self.device_log[3])
            self.meta_size = tf.summary.scalar("end-train memery", self.device_log[4])
            self.all_var_size = tf.summary.scalar("peak memery", self.device_log[5])
            self.gpu_usage = tf.summary.scalar("tf memery usage", self.device_log[6])
            self.peak_gpu_usage = tf.summary.scalar("peak tf memery usage", self.device_log[7])

        self.device_log_summary = tf.summary.merge([
            self.step_duration, self.cpu_usage, self.mem_usage,
            self.total_load, self.meta_size, self.all_var_size,
            self.gpu_usage, self.peak_gpu_usage,
        ])

    @staticmethod
    def _slice_batch_dict(bacth_data, str_idx, end_idx):
        dict_ = {}
        for key in bacth_data:
            dict_[key] = bacth_data[key][str_idx:end_idx]
        return dict_

    @staticmethod
    def _update_dict(dict1, dict2):
        for key in dict2:
            if key in dict1.keys():
                dict1[key].append(dict2[key])
            else:
                dict1[key] = [dict2[key]]

    def compute_gradients(self, loss):
        grads = self.optimizer.compute_gradients(
            loss, var_list=self._get_trainable_vars(),
        )
        grads = self._wrapper_gradients(grads)
        return grads

    def _reconstract_batch_output(self, ):
        raise NotImplementedError

    def _build_backward(self, ):
        _EMA = self.EMA if hasattr(self, 'EMA') else False
        with tf.name_scope("apply_gradients"), tf.device(self.controller):
            avg_grags = average_gradients(self.tower_grads)
        if _EMA:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            op_seg_ = self.optimizer.apply_gradients(avg_grags)
            with tf.control_dependencies([op_seg_]):
                self.optimize = self.ema.apply(self._get_trainable_vars())
        else:
            self.optimize = self.optimizer.apply_gradients(avg_grags)

    def _build_index(self):
        raise NotImplementedError

    def _build_summary(self):
        with tf.device(self.controller):
            self._build_summary_graph()
            self._build_devicelog_graph()

    def build(self):
        force_print('='*20+'Managing Multi-GPUs Process >> START'+'='*20)
        self.controller = "/device:CPU:0"
        sub_batch = self.sub_batch_size
        devices = self.devices
        n_gpus = self.n_gpus
        force_print('Controller:', self.controller)
        force_print('Avaliable GPUs:', devices)
        force_print(' '*4+'Batchsize:{}'.format(self.batch_size)+
                    '= > '+'Total GPUs: {}'.format(n_gpus)+
                    '= > '+'batch per GPU:{}'.format(sub_batch)
                    )
        self.tower_grads = []
        self._build_dataloader()
        self._build_optimizer()
        with tf.variable_scope('MT_GPU') as outer_scope:
            self.train_output_dict_, self.test_output_dict_ = {}, {}
            self.train_loss_dict_, self.test_loss_dict_ = {}, {}

            for i, device_id in enumerate(devices):
                force_print('-- ALLOCATING GRAPHS ON >>', device_id)
                name = 'tower_{}'.format(i)
                with tf.name_scope(name):
                    with tf.device(assign_to_device(device_id, self.controller)):
                        force_print(''*4+'-'*3+'allocated batch {}-{}'.format(i*sub_batch, (i+1)*sub_batch))
                        train_net_outputs = self._build_net(
                            self._slice_batch_dict(self.batched_data_train, i*sub_batch, (i+1)*sub_batch),
                            is_training=True,
                        )
                        test_net_outputs = self._build_net(
                            self._slice_batch_dict(self.batched_data_test, i*sub_batch, (i+1)*sub_batch),
                            is_training=False,
                        )
                        #
                        self._update_dict(self.train_output_dict_, train_net_outputs)
                        self._update_dict(self.test_output_dict_, test_net_outputs)

                    with tf.variable_scope("loss", reuse=tf.AUTO_REUSE), tf.device(self.controller):
                        loss_train = self._get_loss(
                            train_net_outputs,
                            self._slice_batch_dict(self.batched_data_train, i*sub_batch, (i+1)*sub_batch),
                        )
                        loss_test = self._get_loss(
                            test_net_outputs,
                            self._slice_batch_dict(self.batched_data_test, i*sub_batch, (i+1)*sub_batch),
                        )
                        #
                        self._update_dict(self.train_loss_dict_, loss_train)
                        self._update_dict(self.test_loss_dict_, loss_test)
                    with tf.variable_scope("cmpt_grad", reuse=tf.AUTO_REUSE), \
                         tf.device(assign_to_device(device_id, self.controller)):
                        self.tower_grads.append(self.compute_gradients(loss_train['total_loss']))

                outer_scope.reuse_variables()

        # reconstruct output & loss
        self._reconstract_batch_output()

        '''BACKWARD'''
        self._build_backward()

        force_print('='*20+'Managing Multi-GPUs Process >> FINISH'+'='*20+'\n\n')

        '''BuildIndex'''
        self._build_index()

        '''Summary'''
        self._build_summary()

    @staticmethod
    def tf_train_setup():
        run_option = tf.RunOptions(report_tensor_allocations_upon_oom=True,
                                   trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        peak_tf_mem_MB = tf.contrib.memory_stats.MaxBytesInUse()
        loop_tf_mem_MB = tf.contrib.memory_stats.BytesInUse()
        return {
            'run_option': run_option,
            'run_metadata': run_metadata,
            'peak_tf_mem_MB': peak_tf_mem_MB,
            'loop_tf_mem_MB': loop_tf_mem_MB,
        }

    def restore_net_for_train(self, global_step):
        if self.restore_seg_net:
            force_print('>'*20, 'LOAD GLOBAL CHECKPOINT ...')
            trainable_vars = tf.trainable_variables()
            variables_to_restore = [t for t in trainable_vars if t.name.startswith('MT_GPU')]
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(self.sess, self.reload_path)
            force_print('- checkpoint loaded from {},\n - global step set to {}'.format(self.reload_path, global_step))
            force_print('>'*20, 'LOAD GLOBAL CHECKPOINT FINISHED')
        else:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                force_print('>'*20, 'RESTORE LOCAL CHECKPOINT...')
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                _EMA = self.EMA if hasattr(self, 'EMA') else False
                if _EMA:
                    ema_name_map_ = {}
                    for v in self._get_trainable_vars():
                        ema_name_map_[self.ema.average_name(v)] = v
                    saver = tf.train.Saver(ema_name_map_)
                else:
                    saver = tf.train.Saver()
                saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
                global_step = int(ckpt_name.split('-')[-1])
                force_print(' - checkpoint restored from local model,\n - global step set to {}'.format(global_step))
                force_print('>'*20, 'RESTORE LOCAL CHECKPOINT FINISHED')
            else:
                force_print('>'*20, 'NO CHECKPOINT FOUND, START NEW TRAINING')

        return global_step

    def epoch_eva_for_train(self, epoch_eva_for_train):
        raise NotImplementedError

    def train_step(self, log_):
        raise NotImplementedError

    def _devive_log(self, log_):
        global_step = log_['global_step']
        t_per_iter = log_['t_per_iter']

        peak_mem_use = log_['peak_mem_use']
        loop_mem_use = log_['loop_mem_use']

        _RECORD_DEVICE_LOG = self.RECORD_DEVICE_LOG if hasattr(self, 'RECORD_DEVICE_LOG') else True

        total_load, cpu_rate, mem_rate, dur_time = get_top_info()
        mem_now, mem_peak = tracemalloc.get_traced_memory()
        log_list = [t_per_iter, cpu_rate, mem_rate, total_load[0],
                    mem_now//1048576, mem_peak//1048576, peak_mem_use//1048576, loop_mem_use//1048576]
        if _RECORD_DEVICE_LOG:
            log_str = self.sess.run(self.device_log_summary, feed_dict={self.device_log: log_list})
            self.writer.add_summary(log_str, global_step)

    def test_step(self, log_, ):
        raise NotImplementedError

    def _save_debugimgs(self, log_):
        raise NotImplementedError

    def _print_log(self, log_):
        print('No Logs ...')

    def train(self):
        _PRETRAIN_EPOCH = self.PRETRAIN_EPOCH if hasattr(self, 'PRETRAIN_EPOCH') else 15

        force_print('='*30, 'START TRAINING PROCESS', '='*30)
        self.sess.run(tf.global_variables_initializer())
        global_step = 0
        self.saver = tf.train.Saver(max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.tf_info_dict = self.tf_train_setup()

        self._initialize_dataset(init_datasets='TRAIN')

        global_step = self.restore_net_for_train(global_step)

        # avoid adding new nodes
        tf.get_default_graph().finalize()

        strt_ep = global_step//self.iters
        tracemalloc.start()
        for ep in range(strt_ep, self.epoch):
            if ep > _PRETRAIN_EPOCH:
                self.epoch_eva_for_train(global_step)

            self._initialize_dataset(init_datasets='TEST')
            for ir in range(self.iters):
                ts = time.time()
                # print_()
                # print_('Starttiming --')
                global_step += 1

                # log infos pr eiter
                log_ = {}
                log_['global_step'] = global_step
                log_['epoch'] = ep
                log_['iter'] = ir
                # print_('-<timebeforeself.train__step>-%2.2fs'%(time.time()-ts))

                # train step
                tnow = time.time()
                self.train_step(log_)
                # print_('-<timetillself.train_step>-%2.2fs'%(time.time()-ts))

                # devive_log
                log_['t_per_iter'] = time.time()-tnow
                self._devive_log(log_)

                # test step
                if ir%50 == 0:
                    self.test_step(log_)
                    self._save_debugimgs(log_)
                    self._print_log(log_)
                    force_print(time.asctime(time.localtime(time.time())), ''*10, self.model_dir)

                # print_('<<time of entire iter>>-%2.2fs'%(time.time()-ts))

            leak_mem = gc.collect()
            force_print('MEMERY LEAK...', leak_mem)
            self.saver.save(self.sess, os.path.join(self.ckpt_dir, 'segnet.ckpt'), global_step=global_step)

        force_print('='*30, 'TRAIN FINISHED', '='*30, end='\n\n')
        return None

    def test(self):
        raise NotImplementedError

    def evaluate_stage(self):
        raise NotImplementedError


if __name__ == '__main__':
    print('NOT IMPLEMENTED YET')
