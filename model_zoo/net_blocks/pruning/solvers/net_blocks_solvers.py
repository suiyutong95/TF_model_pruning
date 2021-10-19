from pruning_utils.solver import register_prune_op_solver
from pruning_utils.utils import solver_cfg
import numpy as np

from ..net_blocks_for_pruning import conv3D_block, start_77conv_block

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>utils>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>solvers>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@register_prune_op_solver(method_key='weight_mean', block_funcs=[
    conv3D_block,
    start_77conv_block
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
    conv_weights = variables_dict['conv3d/kernel:0'][:,:,:,np.where(in_mask)[0],:]

    # find top channels
    selected_idxes = np.argsort(conv_weights.mean(axis=(0,1,2,3)))[::-1][:int(out_ch*scale)]
    conv_mask = []
    for i in range(out_ch):
        if i in selected_idxes:
            conv_mask.append(1)
        else:
            conv_mask.append(0)

    # prune variables
    conv_weights = conv_weights[:,:,:,:,np.where(conv_mask)[0]]
    conv_bias = variables_dict['conv3d/bias:0'][np.where(conv_mask)[0],]
    # updated dict
    variables_dict['conv3d/kernel:0'] = conv_weights
    variables_dict['conv3d/bias:0'] = conv_bias

    # apply to norm func
    if 'GroupNorm/beta:0' in variables_dict.keys():
        # prune variables
        gn_beta = variables_dict['GroupNorm/beta:0'][np.where(conv_mask)[0]]
        gn_gamma = variables_dict['GroupNorm/gamma:0'][np.where(conv_mask)[0]]
        # updated dict
        variables_dict['GroupNorm/beta:0'] = gn_beta
        variables_dict['GroupNorm/gamma:0'] = gn_gamma
    elif 'BatchNorm/beta:0' in variables_dict.keys():
        # prune variables
        bn_beta = variables_dict['BatchNorm/beta:0'][np.where(conv_mask)[0]]
        bn_gamma = variables_dict['BatchNorm/gamma:0'][np.where(conv_mask)[0]]
        bn_mean = variables_dict['BatchNorm/moving_mean:0'][np.where(conv_mask)[0]]
        bn_var = variables_dict['BatchNorm/moving_variance:0'][np.where(conv_mask)[0]]
        # updated dict
        variables_dict['BatchNorm/beta:0'] = bn_beta
        variables_dict['BatchNorm/gamma:0'] = bn_gamma
        variables_dict['BatchNorm/moving_mean:0'] = bn_mean
        variables_dict['BatchNorm/moving_variance:0'] = bn_var
    else:
        raise ValueError('only [BN, GN] is pruning-supported, yet')

    # update params
    if 'channels' in pnode.block_func_kwargs.keys():
        pnode.block_func_kwargs['channels'] = sum(conv_mask)
    else:
        pnode.block_func_args[0] = sum(conv_mask)

    return conv_mask, variables_dict








