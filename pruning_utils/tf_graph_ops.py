import tensorflow as tf


def get_ops_between_tensors(tensor_in, tensor_out, scope=''):
    '''
    {find tf ops between tensor_in and tensor_out with recursion method}
    '''
    all_tensors = []
    all_ops = []
    _searched_tensors = []

    def rc(tensor, tensors, ops):
        # return when tensor was searched
        if tensor.name in _searched_tensors:
            return
        _searched_tensors.append(tensor.name)

        if tensor == tensor_out:
            for x in tensors:
                if x not in all_tensors: all_tensors.append(x)
            for x in ops:
                if x not in all_ops: all_ops.append(x)
            return

        # run recursion
        for op in tensor.consumers():
            if scope not in op.name:
                continue
            if not op.outputs:  ## op.outputs == []
                return
            else:
                for ts in op.outputs:
                    rc(ts, tensors+[ts.name], ops+[op.name])

    if tensor_in == tensor_out:
        raise ValueError('tensor_in and tensor_out cannot be same tensor')

    rc(tensor_in, [], [])

    # ops cannot be empty
    if not all_ops: raise ValueError('tensor_out is not a child of tensor_in')

    return all_ops, all_tensors
