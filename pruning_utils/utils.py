import tensorflow as tf
from functools import wraps
from pruning_utils.tf_graph_ops import get_ops_between_tensors
import pickle
import copy


class P_graph():
    '''
    {graph for pruning}
    '''

    def __init__(self, pnode, basescope):
        self.head_node = pnode
        self.all_nodes = [pnode]
        self.output_nodes = {}
        self.base_scope = basescope
        # pnode.graph = self

    def print_info(self):
        print('Total number of nodes  -> {}'.format(len(self.all_nodes)))
        print('Number of output nodes -> {}'.format(len(self.output_nodes)))
        for i, (k, v) in enumerate(zip(self.output_nodes.keys(), self.output_nodes.values())):
            print(
                '    {}th output key ->  '.format(i), k.ljust(20, ' '), v.scope_id
            )


class P_node():
    '''
    {node for pruning}
    '''

    def __init__(self, x, y, is_head=False, block_name=None):
        '''
        {}
        Input:
            x[P_node]:
            y[tf.Tensor]:
        '''
        self.children = []
        self.parents = None
        self.block_name = block_name
        self.ch_op_type = None
        self.is_head = False
        self.tensor_rebuild_out = None
        if is_head:
            self.is_head = True
            self.tensor_in = x
            self.graph = P_graph(self,('/').join(x.name.split('/')[:-1])+'/')
            self.tensor_out = x
            self.tensor_out_shape = x.shape

        elif type(x) == list:  # add/concat...
            self.tensor_in = [pn.tensor_out for pn in x]
            self.parents = x
            self.graph = x[0].graph
            self.graph.all_nodes.append(self)
            for pn in x: pn.children.append(self)
            self.tensor_out = y
            self.tensor_out_shape = y.shape
        else:
            self.tensor_in = x.tensor_out
            self.parents = x
            self.graph = x.graph
            self.graph.all_nodes.append(self)
            x.children.append(self)
            self.tensor_out = y
            self.tensor_out_shape = y.shape

        self.tf_ops = None
        self.common_scope = None
        self.block_scope = None
        if self.is_head:
            self.scope_id = '__HEAD__'
        else:
            self.scope_id = None
        self.block_func_args = None
        self.block_func_kwargs = None
        self.is_output = False

        # pruning params
        self.pruned_mask = None

    def __add__(self, other):
        with tf.variable_scope(None, 'pn_add', ):
            y = self.tensor_out+other.tensor_out
        pn = P_node([self, other], y)
        pn.block_name = 'pn_add'
        pn.block_func_args = []
        pn.block_func_kwargs = {}
        pn.find_ops()
        return pn

    @staticmethod
    def c__add__(pn_list, _block_scope='pn_add'):
        pn1, pn2 = pn_list
        if pn1.tensor_out.shape[-1] != pn2.tensor_out.shape[-1]:
            raise ValueError(
                'input dims un-match with {} and {}, check graph structure and pruning '
                'configuration'.format(pn1.tensor_out.shape[-1], pn2.tensor_out.shape[-1])
            )
        with tf.variable_scope(None, _block_scope, ):
            y = pn1.tensor_out+pn2.tensor_out
        pn = P_node([pn1, pn2], y)
        pn.block_name = 'pn_add'
        pn.block_func_args = []
        pn.block_func_kwargs = {}
        pn.find_ops()
        return pn

    def __mul__(self, other):
        with tf.variable_scope(None, 'pn_mul', ):
            y = self.tensor_out*other.tensor_out
        pn = P_node([self, other], y)
        pn.block_name = 'pn_mul'
        pn.block_func_args = []
        pn.block_func_kwargs = {}
        pn.find_ops()
        return pn

    @staticmethod
    def c__mul__(pn_list, _block_scope='pn_mul'):
        pn1, pn2 = pn_list
        with tf.variable_scope(None, _block_scope, ):
            y = pn1.tensor_out*pn2.tensor_out
        pn = P_node([pn2, pn2], y)
        pn.block_name = 'pn_mul'
        pn.block_func_args = []
        pn.block_func_kwargs = {}
        pn.find_ops()
        return pn

    def find_ops(self):
        '''
        {}
        '''
        ts_out_name = self.tensor_out.name
        block_suffix = ts_out_name.split(self.block_name)[-1].split('/')[0]
        block_scope = self.block_name+block_suffix
        common_scope = ts_out_name.split(block_scope)[0]

        if type(self.tensor_in) == list:
            op_names, tensor_names = get_ops_between_tensors(self.tensor_in[0], self.tensor_out, block_scope)
        else:
            op_names, tensor_names = get_ops_between_tensors(self.tensor_in, self.tensor_out, block_scope)
        self.tf_ops = op_names
        self.common_scope = common_scope
        self.block_scope = block_scope
        self.scope_id = common_scope+block_scope
        self.scope_id = self.scope_id.replace(self.graph.base_scope,'')


    def as_output(self, key=None):
        self.is_output = True
        self.output_key = key if key is not None else self.scope_id
        self.graph.output_nodes[key] = self

    # def find_related_variables(self):
    #     '''
    #     {}
    #     '''
    #     return None


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> save graph >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def save_graph(graph, path):
    tsin = []
    tsout = []
    tsrout = []
    tsop = []
    tsg = []
    for node in graph.all_nodes:
        tsin.append(node.tensor_in); del node.tensor_in
        tsout.append(node.tensor_out); del node.tensor_out
        tsrout.append(node.tensor_rebuild_out); del node.tensor_rebuild_out
        tsop.append(node.tf_ops); del node.tf_ops
        tsg.append(node.graph); del node.graph

    with open(path, 'wb') as f:
        pickle.dump(graph, f)

    for i, node in enumerate(graph.all_nodes):
        node.tensor_in = tsin[i]
        node.tensor_out = tsout[i]
        node.tensor_rebuild_out = tsrout[i]
        node.tf_ops = tsop[i]
        node.graph = tsg[i]

    return


def load_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> pruning wrapper >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

func_map = {
    'pn_mul': P_node.c__mul__,
    'pn_add': P_node.c__add__,
}

solver_cfg = {
    'pn_mul': {
        'allow_prune': False,
        'ch_op_type': 'element_wise',
        'prune_solver': {},
    },
    'pn_add': {
        'allow_prune': False,
        'ch_op_type': 'element_wise',
        'prune_solver': {},
    },
}


def pruning_wrapper(ch_op_type='single', allow_prune=False):
    def _pruning_wrapper(block_func):
        @wraps(block_func)
        def _wrapper(x, *args, **kwargs):
            if type(x) == tf.Tensor:  ## use for re-build ?
                raise TypeError(
                    'For <tf.Tensor> as input, use <model_zoo.net_blocks> instead of <model_zoo.net_blocks.pruning>')
                # return block_func(x, *args, **kwargs)
            elif type(x) == list:
                y = block_func([pn.tensor_out for pn in x], *args, **kwargs)
            elif type(x) == P_node:
                y = block_func(x.tensor_out, *args, **kwargs)
            else:
                raise TypeError('Un-known type by func <pruning_wrapper>')
            pn = P_node(x, y, block_name=block_func.__name__)
            pn.block_func_args = list(args)
            pn.block_func_kwargs = kwargs
            pn.find_ops()
            return pn

        # register new func
        if block_func.__name__ not in func_map.keys():
            func_map[block_func.__name__] = _wrapper
        if block_func.__name__ not in solver_cfg.keys():
            solver_cfg[block_func.__name__] = {
                'allow_prune': allow_prune,
                'ch_op_type': ch_op_type,
                'prune_solver': {},
            }

        return _wrapper

    return _pruning_wrapper
