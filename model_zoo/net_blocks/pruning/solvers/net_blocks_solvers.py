from pruning_utils.solver import register_prune_op_solver
from pruning_utils.utils import solver_cfg
import numpy as np

from ..net_blocks_for_pruning import conv3D_block, start_77conv_block, upsample3D_block, SE_block_3d


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>utils>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def conv_pruner(chin_mask, chout_mask, variables_dict, scope='conv3d'):
    # prune variables
    conv_weights = variables_dict[scope+'/kernel:0'][:, :, :, np.where(chin_mask)[0], :]
    conv_weights = conv_weights[:, :, :, :, np.where(chout_mask)[0]]
    conv_bias = variables_dict[scope+'/bias:0'][np.where(chout_mask)[0],]
    # updated dict
    variables_dict[scope+'/kernel:0'] = conv_weights
    variables_dict[scope+'/bias:0'] = conv_bias
    return


def gn_pruner(ch_mask, variables_dict, scope='GroupNorm'):
    # prune variables
    gn_beta = variables_dict[scope+'/beta:0'][np.where(ch_mask)[0]]
    gn_gamma = variables_dict[scope+'/gamma:0'][np.where(ch_mask)[0]]
    # updated dict
    variables_dict[scope+'/beta:0'] = gn_beta
    variables_dict[scope+'/gamma:0'] = gn_gamma
    return


def bn_pruner(ch_mask, variables_dict, scope='BatchNorm'):
    # prune variables
    bn_beta = variables_dict[scope+'/beta:0'][np.where(ch_mask)[0]]
    bn_gamma = variables_dict[scope+'/gamma:0'][np.where(ch_mask)[0]]
    bn_mean = variables_dict[scope+'/moving_mean:0'][np.where(ch_mask)[0]]
    bn_var = variables_dict[scope+'/moving_variance:0'][np.where(ch_mask)[0]]
    # updated dict
    variables_dict[scope+'/beta:0'] = bn_beta
    variables_dict[scope+'/gamma:0'] = bn_gamma
    variables_dict[scope+'/moving_mean:0'] = bn_mean
    variables_dict[scope+'/moving_variance:0'] = bn_var
    return


def fc_pruner(chin_mask, chout_mask, variables_dict, scope='dense'):
    # prune variables
    kernel = variables_dict[scope+'/kernel:0'][np.where(chin_mask)[0], :]
    kernel = kernel[:, np.where(chout_mask)[0]]
    bias = variables_dict[scope+'/bias:0'][np.where(chout_mask)[0]]
    # updated dict
    variables_dict[scope+'/kernel:0'] = kernel
    variables_dict[scope+'/bias:0'] = bias
    return


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>solvers>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@register_prune_op_solver(method_key='weight_mean', block_funcs=[
    conv3D_block,
    start_77conv_block,
    upsample3D_block,
])
def conv_prune_wmean(pnode, prune_cfg, variables_dict):
    '''
    {fixed function format}
    PARAMS:
        pnode[P_node]:
        prune_cfg[dict]:
        vairables_dict[dict]:
    RETURNS:
        output_mask[list]: output channel mask
        pruned_variables_dict[dict]:
    '''

    in_mask = pnode.input_mask
    scale = prune_cfg['scale']

    out_ch = variables_dict['conv3d/kernel:0'].shape[-1]

    # read variable
    conv_weights = variables_dict['conv3d/kernel:0'][:, :, :, np.where(in_mask)[0], :]

    # find top channels / mask
    selected_idxes = np.argsort(conv_weights.mean(axis=(0, 1, 2, 3)))[::-1][:int(out_ch*scale)]
    conv_mask = []
    for i in range(out_ch):
        if i in selected_idxes:
            conv_mask.append(1)
        else:
            conv_mask.append(0)

    # pruning weights
    conv_pruner(in_mask, conv_mask, variables_dict, scope='conv3d')
    if 'GroupNorm/beta:0' in variables_dict.keys():
        gn_pruner(conv_mask, variables_dict, scope='GroupNorm')
    elif 'BatchNorm/beta:0' in variables_dict.keys():
        bn_pruner(conv_mask, variables_dict, scope='BatchNorm')
    else:
        raise ValueError('only [BN, GN] is pruning-supported, yet')

    # update params
    if 'channels' in pnode.block_func_kwargs.keys():
        pnode.block_func_kwargs['channels'] = sum(conv_mask)
    else:
        pnode.block_func_args[0] = sum(conv_mask)

    return conv_mask, variables_dict


@register_prune_op_solver(method_key='weight_mean', block_funcs=[
    SE_block_3d
])
def se_prune_wmean(pnode, prune_cfg, variables_dict):
    '''
    {fixed function format}
    PARAMS:
        pnode[P_node]:
        prune_cfg[dict]:
        vairables_dict[dict]:
    RETURNS:
        output_mask[list]: output channel mask
        pruned_variables_dict[dict]:
    '''

    in_mask = pnode.input_mask
    ch_in = len(in_mask)
    ch_inp = sum(in_mask)
    ch_1 = variables_dict['dense/kernel:0'].shape[-1]
    ratio = len(in_mask)//ch_1
    if ch_inp//ratio == 0:
        ratio = ch_inp
        ch_1p = 1
    else:
        ch_1p = ch_inp//ratio

    #
    fc1_weights = variables_dict['dense/kernel:0'][np.where(in_mask)[0], :]
    selected_idxes = np.argsort(fc1_weights.mean(axis=(0)))[::-1][:ch_1p]
    fc1_mask = []
    for i in range(ch_1):
        if i in selected_idxes:
            fc1_mask.append(1)
        else:
            fc1_mask.append(0)

    #
    fc2_weights = variables_dict['dense_1/kernel:0'][np.where(fc1_mask)[0], :]
    selected_idxes = np.argsort(fc2_weights.mean(axis=(0)))[::-1][:ch_inp]
    fc2_mask = []
    for i in range(ch_in):
        if i in selected_idxes:
            fc2_mask.append(1)
        else:
            fc2_mask.append(0)

    # pruning weights
    fc_pruner(in_mask, fc1_mask, variables_dict, scope='dense')
    fc_pruner(fc1_mask, fc2_mask, variables_dict, scope='dense_1')

    # update params
    if 'ratio' in pnode.block_func_kwargs.keys():
        pnode.block_func_kwargs['ratio'] = ratio
    else:
        pnode.block_func_args[0] = ratio

    return fc2_mask, variables_dict
