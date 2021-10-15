import tensorflow as tf
from .utils import P_node, func_map


def rebuild_tf_graph(tf_in, graph):
    """
    {rebuild tf model graph}
    Input:
        tf_in[tf.Tensor]:
        graph[P_graph]: pruning graph
    Return:
        graph_new[P_graph]: new graph rebuilded
    """
    nodes_map = {}
    head_node = P_node(tf_in, y=None, is_head=True, ch_op_type=None)

    for i, pn_old in enumerate(graph.all_nodes):
        if pn_old.is_head:
            nodes_map[pn_old.scope_id] = head_node
        else:
            if type(pn_old.parents) != list:
                if '_block_scope' in pn_old.block_func_kwargs.keys():
                    del pn_old.block_func_kwargs['_block_scope']
                nodes_map[pn_old.scope_id] = func_map[pn_old.block_name](
                    nodes_map[pn_old.parents.scope_id],
                    _block_scope=pn_old.scope_id,
                    *pn_old.block_func_args,
                    **pn_old.block_func_kwargs,
                )
                if pn_old.is_output:
                    nodes_map[pn_old.scope_id].as_output(pn_old.output_key)
            else:
                if '_block_scope' in pn_old.block_func_kwargs.keys():
                    del pn_old.block_func_kwargs['_block_scope']
                nodes_map[pn_old.scope_id] = func_map[pn_old.block_name](
                    [nodes_map[x.scope_id] for x in pn_old.parents],
                    _block_scope=pn_old.scope_id,
                    *pn_old.block_func_args,
                    **pn_old.block_func_kwargs
                )
                if pn_old.is_output:
                    nodes_map[pn_old.scope_id].as_output(pn_old.output_key)

    return head_node.graph


def rebuild_tf_graph_rc(tf_in, graph):
    """
    {rebuild tf model graph with recursion method}
    Input:
        tf_in[tf.Tensor]:
        graph[P_graph]: pruning graph
    Return:
        graph_new[P_graph]: new graph rebuilded
    """

    def rc(pnode, new_pnode_p):
        """
        Input:
            pnode[]: current node from old graph
            new_pnode_p[]: parents node in new graph
        """
        if pnode.scope_id in nodes_map.keys(): return  # skip repeated nodes

        if pnode.is_head:
            nodes_map[pnode.scope_id] = new_pnode_p
            for pn in pnode.children:
                rc(pn, new_pnode_p)
        elif type(pnode.parents) != list:
            block_func = func_map[pnode.block_name]
            pkwargs = pnode.block_func_kwargs
            pkwargs['_block_scope'] = pnode.scope_id
            new_pnode_c = block_func(new_pnode_p, *pnode.block_func_args, **pkwargs)
            if pnode.is_output:
                new_pnode_c.as_output(pnode.output_key)
            nodes_map[pnode.scope_id] = new_pnode_c
            if not pnode.children: return  # return when node have no children
            for pn in pnode.children:
                rc(pn, new_pnode_c)
        else:
            if all([p.scope_id in nodes_map.keys() for p in pnode.parents]):
                block_func = func_map[pnode.block_name]
                pkwargs = pnode.block_func_kwargs
                pkwargs['_block_scope'] = pnode.scope_id
                new_pnode_c = block_func([nodes_map[p.scope_id] for p in pnode.parents],
                                         *pnode.block_func_args, **pkwargs)
                if pnode.is_output:
                    new_pnode_c.as_output(pnode.output_key)
                nodes_map[pnode.scope_id] = new_pnode_c
                if not pnode.children: return  # return when node have no children
                for pn in pnode.children:
                    rc(pn, new_pnode_c)
            else:
                return

    # run recursion
    nodes_map = {}
    head_node = P_node(tf_in, y=None, is_head=True, ch_op_type=None)
    rc(graph.head_node, head_node)

    return head_node.graph
