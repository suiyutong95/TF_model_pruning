import tensorflow as tf
from functools import wraps
from pruning_utils.tf_graph_ops import get_ops_between_tensors



class P_graph():
    '''
    {graph for pruning}
    '''
    def __init__(self, pnode):
        self.head_node = pnode
        self.all_nodes = [pnode]
        self.output_nodes = {}
        # pnode.graph = self

    def print_info(self):
        print('Total number of nodes  -> {}'.format(len(self.all_nodes)))
        print('Number of output nodes -> {}'.format(len(self.output_nodes)))
        for i, (k, v) in enumerate(zip(self.output_nodes.keys(),self.output_nodes.values())):
            print(
                '    {}th output key ->  '.format(i), k.ljust(20,' '), v.scope_id
            )

class P_node():
    '''
    {node for pruning}
    '''

    def __init__(self, x, y, is_head=False, ch_op_type='scale', block_name=None):
        '''
        {}
        Input:
            x[P_node]:
            y[tf.Tensor]:
        '''
        self.children = []
        self.parents = None
        self.block_name = block_name
        self.ch_op_type = 'scale'
        self.is_head = False
        self.tensor_rebuild_out = None
        if is_head:
            self.is_head = True
            self.tensor_in = x
            self.graph = P_graph(self)
            self.tensor_out = x
        elif type(x) == list:  # add/concat...
            self.tensor_in = [pn.tensor_out for pn in x]
            self.parents = x
            self.graph = x[0].graph
            self.graph.all_nodes.append(self)
            for pn in x: pn.children.append(self)
            self.tensor_out = y
        else:
            self.tensor_in = x.tensor_out
            self.parents = x
            self.graph = x.graph
            self.graph.all_nodes.append(self)
            x.children.append(self)
            self.tensor_out = y
        if ch_op_type == 'scale':
            self.ch_scale = int(y.shape[-1])/int(x.tensor_out.shape[-1])

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

    def __add__(self, other):
        with tf.variable_scope(None, 'pn_add',):
            y = self.tensor_out + other.tensor_out
        pn = P_node([self, other], y, ch_op_type='same')
        pn.block_name = 'pn_add'
        pn.block_func_args = []
        pn.block_func_kwargs = {}
        pn.find_ops()
        return pn

    @staticmethod
    def c__add__(pn_list, _block_scope='pn_add'):
        pn1, pn2 = pn_list
        with tf.variable_scope(None, _block_scope,):
            y = pn1.tensor_out + pn2.tensor_out
        pn = P_node([pn1, pn2], y, ch_op_type='same')
        pn.block_name = 'pn_add'
        pn.block_func_args = []
        pn.block_func_kwargs = {}
        pn.find_ops()
        return pn

    def __mul__(self, other):
        with tf.variable_scope(None, 'pn_mul',):
            y = self.tensor_out * other.tensor_out
        pn = P_node([self, other], y, ch_op_type='same')
        pn.block_name = 'pn_mul'
        pn.block_func_args = []
        pn.block_func_kwargs = {}
        pn.find_ops()
        return pn

    @staticmethod
    def c__mul__(pn_list, _block_scope='pn_mul'):
        pn1, pn2 = pn_list
        with tf.variable_scope(None, _block_scope,):
            y = pn1.tensor_out * pn2.tensor_out
        pn = P_node([pn2, pn2], y, ch_op_type='same')
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
        block_scope = self.block_name + block_suffix
        common_scope = ts_out_name.split(block_scope)[0]

        if type(self.tensor_in) == list:
            op_names, tensor_names = get_ops_between_tensors(self.tensor_in[0], self.tensor_out, block_scope)
        else:
            op_names, tensor_names = get_ops_between_tensors(self.tensor_in, self.tensor_out, block_scope)
        self.tf_ops = op_names
        self.common_scope = common_scope
        self.block_scope = block_scope
        self.scope_id = common_scope + block_scope

    def as_output(self,key=None):
        self.is_output = True
        self.output_key = key if key is not None else self.scope_id
        self.graph.output_nodes[key] = self



    # def find_related_variables(self):
    #     '''
    #     {}
    #     '''
    #     return None

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> pruning wrapper >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

func_map = {
    'pn_mul': P_node.c__mul__,
    'pn_add': P_node.c__add__,
}

def pruning_wrapper(pruning_on=True, ch_op_type='scale'):
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
            pn = P_node(x, y, ch_op_type=ch_op_type, block_name=block_func.__name__)
            pn.block_func_args = args
            pn.block_func_kwargs = kwargs
            pn.find_ops()
            return pn

        # register new func
        if block_func.__name__ not in func_map.keys():
            func_map[block_func.__name__] = _wrapper

        return _wrapper

    return _pruning_wrapper
