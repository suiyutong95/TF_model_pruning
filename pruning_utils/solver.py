from .utils import solver_cfg
from functools import wraps
import copy

solver_lib = {
    'ch_op': {},
    'prune_op': {},
}


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>@wrapper>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# register channel operation wrapper
def register_ch_op_solver(key):
    def _register_sovler(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if key in solver_lib['ch_op'].keys():
            raise ValueError('ch_op key [{}] is already registered. Try a new key'.format(key))

        solver_lib['ch_op'][key] = func

        return _wrapper

    return _register_sovler


# register prune operation wrapper
def register_prune_op_solver(method_key, block_funcs):
    def _register_sovler(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if method_key not in solver_lib['prune_op'].keys():
            solver_lib['prune_op'][method_key] = {}
        for bkf in block_funcs:
            if bkf.__name__ in solver_lib['prune_op'][method_key].keys():
                raise KeyError('method already registered')
            if bkf.__name__ not in solver_cfg.keys():
                raise KeyError('block name not registered')
            if method_key in solver_cfg[bkf.__name__]['prune_solver'].keys():
                raise KeyError('block method crashed')

            solver_lib['prune_op'][method_key][bkf.__name__] = func
            solver_cfg[bkf.__name__]['prune_solver'][method_key] = (method_key, bkf.__name__)

        return _wrapper

    return _register_sovler


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# update pnodes and weights
def prune_solver(pnode, weights, pruning_cfg):
    print('   - pruning node <{}> ...'.format(pnode.scope_id).ljust(50, ' '), end=' ')
    _s_cfg = solver_cfg[pnode.block_name]

    _ch_op_lib = solver_lib['ch_op']
    _pn_op_lib = solver_lib['prune_op']

    # get weights
    w_dict = {}
    for k, v in weights.items():
        k_ = k[len(pnode.graph.base_scope):]
        if k_[:len(pnode.scope_id)+1] == pnode.scope_id+'/':
            w_dict[k_[len(pnode.scope_id)+1:]] = v

    # channel ops
    if (type(pnode.parents) != list) and (pnode.parents.is_head):
        input_mask = [1 for _ in range(pnode.parents.tensor_out_shape[-1])]
    else:
        chop_func = _ch_op_lib[_s_cfg['ch_op_type']]
        input_mask = chop_func(pnode)

    # return when not need to pruning
    if not _s_cfg['allow_prune']:
        pnode.output_mask = input_mask
        print('- SKIPED <prune unable>')
        return

    # pruning
    p_solvers_cfg = _s_cfg['prune_solver']
    if not p_solvers_cfg:
        print('- SKIPED <prune solver not implemented yet>')
        pnode.output_mask = input_mask
        return
    pn_method = pruning_cfg[pnode.scope_id]['method']
    if pn_method not in p_solvers_cfg.keys():
        raise KeyError('cannot found related prune method <{}>, check key or whether registered'
                       ' in <pruning_utils.solver_cfg>'.format(pn_method))

    pnode.input_mask = input_mask
    prune_func = _pn_op_lib[p_solvers_cfg[pn_method][0]][p_solvers_cfg[pn_method][1]]
    pnode.output_mask, w_dict = prune_func(pnode, pruning_cfg[pnode.scope_id], w_dict)

    # update weights
    for k, v in w_dict.items():
        weights[pnode.graph.base_scope+pnode.scope_id+'/'+k] = v

    print('- DONE')
    return


# prune entire gragh
def prune_though_gragh(graph, weight_dict, pruning_cfg):
    print('Start pruning though entire graph ...')

    for pn in graph.all_nodes:
        if pn.is_head:
            continue
        prune_solver(pn, weight_dict, pruning_cfg)

    print('Prune Done!')
