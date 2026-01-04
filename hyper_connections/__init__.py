from hyper_connections.hyper_connections import (
    HyperConnections,
    get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions,
    Residual,
    StreamEmbed,
    AttentionPoolReduceStream
)

# export with mc prefix, as well as mHC

from hyper_connections.manifold_constrained_hyper_connections import (
    ManifoldConstrainedHyperConnections,
    get_expand_reduce_stream_functions as mc_get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions as mc_get_init_and_expand_reduce_stream_functions
)

mHC = ManifoldConstrainedHyperConnections
