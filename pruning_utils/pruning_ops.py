# import tensorflow as tf
# from .utils import P_node, func_map
#
#
# def prune_graph(tf_in, graph):
#     """
#     {rebuild tf model graph}
#     Input:
#         tf_in[tf.Tensor]:
#         graph[P_graph]: pruning graph
#     Return:
#         graph_new[P_graph]: new graph rebuilt
#     """
#     nodes_map = {}
#     head_node = P_node(tf_in, y=None, is_head=True)
#
#     for i, pn_old in enumerate(graph.all_nodes):
#         if pn_old.is_head:
#             nodes_map[pn_old.scope_id] = head_node
#         else:
#             if type(pn_old.parents) != list:
#                 if '_block_scope' in pn_old.block_func_kwargs.keys():
#                     del pn_old.block_func_kwargs['_block_scope']
#                 nodes_map[pn_old.scope_id] = func_map[pn_old.block_name](
#                     nodes_map[pn_old.parents.scope_id],
#                     _block_scope=pn_old.scope_id,
#                     *pn_old.block_func_args,
#                     **pn_old.block_func_kwargs,
#                 )
#                 if pn_old.is_output:
#                     nodes_map[pn_old.scope_id].as_output(pn_old.output_key)
#             else:
#                 if '_block_scope' in pn_old.block_func_kwargs.keys():
#                     del pn_old.block_func_kwargs['_block_scope']
#                 nodes_map[pn_old.scope_id] = func_map[pn_old.block_name](
#                     [nodes_map[x.scope_id] for x in pn_old.parents],
#                     _block_scope=pn_old.scope_id,
#                     *pn_old.block_func_args,
#                     **pn_old.block_func_kwargs
#                 )
#                 if pn_old.is_output:
#                     nodes_map[pn_old.scope_id].as_output(pn_old.output_key)
#
#     return head_node.graph
#
