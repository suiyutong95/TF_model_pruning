from pruning_utils.solver import register_ch_op_solver

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>solvers>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@register_ch_op_solver('single')
def ch_op_single(pnode):
    return pnode.parents.output_mask


@register_ch_op_solver('concat')
def ch_op_concat(pnode):
    input_mask = []
    for x in pnode.parents:
        input_mask.extend(pnode.parents.output_mask)
    return input_mask


@register_ch_op_solver('element_wise')
def ch_op_element_wise(pnode):
    '''
    {develope this in the future}
    '''
    # print('channel operation for "element_wise" has not developed !!!')
    return pnode.parents[0].output_mask