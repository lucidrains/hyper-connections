import pytest
param = pytest.mark.parametrize

import torch
from torch import nn

@param('num_fracs', (1, 4))
@param('disable', (False, True))
@param('manifold_constrained', (False, True))
def test_readme(
    num_fracs,
    disable,
    manifold_constrained
):

    # a single branch layer

    branch = nn.Linear(512, 512)

    # before

    residual = torch.randn(2, 1024, 512)

    residual = branch(residual) + residual

    # after, say 4 streams in paper

    if manifold_constrained:
        from hyper_connections.manifold_constrained_hyper_connections import get_init_and_expand_reduce_stream_functions
    else:
        from hyper_connections import get_init_and_expand_reduce_stream_functions

    init_hyper_conn, expand_stream, reduce_stream = get_init_and_expand_reduce_stream_functions(4, num_fracs = num_fracs, disable = disable)

    # 1. wrap your branch function

    hyper_conn_branch = init_hyper_conn(dim = 512, branch = branch)

    # 2. expand to 4 streams, this must be done before your trunk, typically a for-loop with many branch functions

    residual = expand_stream(residual)

    # 3. forward your residual as usual into the wrapped branch function(s)

    residual = hyper_conn_branch(residual) 

    # 4. reduce 4 streams with a summation, this has to be done after your for-loop trunk. for transformer, unsure whether to do before or after final norm

    residual = reduce_stream(residual)

    assert residual.shape == (2, 1024, 512)

def test_manual():
    # a single branch layer

    branch = nn.Linear(512, 512)

    # before

    residual = torch.randn(2, 1024, 512)

    residual = branch(residual) + residual

    # after, say 4 streams in paper

    from hyper_connections import get_init_and_expand_reduce_stream_functions

    init_hyper_conn, expand_stream, reduce_stream = get_init_and_expand_reduce_stream_functions(4)

    # 1. instantiate hyper connection with correct number of streams (4 in this case) - or use the init function above

    hyper_conn = init_hyper_conn(dim = 512)

    # 2. expand to 4 streams

    residual = expand_stream(residual)

    # 3. forward your residual into hyper connection for the branch input + add residual function (learned betas)

    branch_input, add_residual = hyper_conn(residual)

    branch_output = branch(branch_input)

    residual = add_residual(branch_output)

    # or you can do it in one line as so -> residual = hyper_conn.decorate_branch(branch)(residual)

    # 4. reduce 4 streams with a summation, this has to be done after your for loop trunk

    residual = reduce_stream(residual)
    assert residual.shape == (2, 1024, 512)

@pytest.mark.parametrize('disable', (False, True))
def test_multi_input_hyper_connections(disable):

    # two branch layers

    class CustomModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
            self.second_linear = nn.Linear(256, 512)
            self.third_linear = nn.Linear(128, 512)

        def forward(self, x, second, *, third):
            return self.linear(x) + self.second_linear(second) + self.third_linear(third), 3.

    branch = CustomModule()

    # before

    residual = torch.randn(3, 1024, 512)
    second_residual = torch.randn(3, 1024, 256)
    third_residual = torch.randn(3, 1024, 128)

    # residual = branch1(residual) + branch2(residual) + residual

    # after, say 4 streams in paper

    from hyper_connections.hyper_connections_with_multi_input_streams import HyperConnections

    init_hyper_conn, expand_stream, reduce_stream = HyperConnections.get_init_and_expand_reduce_stream_functions(4, disable = disable)

    # 1. instantiate hyper connection with correct number of streams (4 in this case) - or use the init function above

    hyper_conn = init_hyper_conn(
        dim = 512,
        branch = branch,
        additional_input_paths = [
            (1, 256),        # points at second residual stream, first arg
            ('third', 128)   # points at third residual stream, keyword argument 'third'
        ],
        layer_index = 1,
    )

    # 2. expand to 4 streams

    residual = expand_stream(residual)
    second_residual = expand_stream(second_residual)
    third_residual = expand_stream(third_residual)

    # 3. forward your residual into hyper connection for the branch input + add residual function (learned betas)

    residual, rest_output = hyper_conn(residual, second_residual, third = third_residual)

    residual = reduce_stream(residual)

    assert residual.shape == (3, 1024, 512)

@pytest.mark.parametrize('disable', (False, True))
def test_channel_first_hyper_connection(disable):

    # a single branch layer

    branch = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding = 1),
        nn.SiLU(),
        nn.Conv2d(512, 512, 3, padding = 1)
    )

    # before

    residual = torch.randn(2, 512, 16, 16)

    before_residual = branch(residual) + residual

    # after, say 4 streams in paper

    from hyper_connections.hyper_connections_channel_first import get_init_and_expand_reduce_stream_functions

    init_hyper_conn, expand_stream, reduce_stream = get_init_and_expand_reduce_stream_functions(4, disable = disable)

    # 1. wrap your branch function

    hyper_conn_branch = init_hyper_conn(dim = 512, branch = branch)

    # 2. expand to 4 streams, this must be done before your trunk, typically a for-loop with many branch functions

    residual = expand_stream(residual)

    # 3. forward your residual as usual into the wrapped branch function(s)

    residual = hyper_conn_branch(residual) 

    # 4. reduce 4 streams with a summation, this has to be done after your for-loop trunk. for transformer, unsure whether to do before or after final norm

    after_residual = reduce_stream(residual)

    assert before_residual.shape == after_residual.shape

def test_mhc_dtype_restoration():
    from hyper_connections.manifold_constrained_hyper_connections import ManifoldConstrainedHyperConnections

    mhc = ManifoldConstrainedHyperConnections(
        num_residual_streams = 4,
        dim = 64,
        add_branch_out_to_residual = True
    )

    residual = torch.randn(4, 1, 64).half()

    branch_input, _, residual_kwargs = mhc.width_connection(residual)

    assert branch_input.dtype == torch.half
    assert residual_kwargs['beta'].dtype == torch.half

    branch_output = torch.randn_like(branch_input).half()
    residual = mhc.depth_connection(branch_output, residual, **residual_kwargs)

    assert residual.dtype == torch.half

@param('num_dynamic_alpha_proposals', (1, 2))
def test_mhc_vit(
    num_dynamic_alpha_proposals
):

    from hyper_connections.vit import ViT

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        num_residual_streams = 4,
        num_dynamic_alpha_proposals = num_dynamic_alpha_proposals
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img) # (1, 1000)
    assert preds.shape == (1, 1000)

@param('num_fracs', (1, 2))
@param('num_streams', (1, 3))
@param('disable', (False, True))
@param('add_attn_pool_reduce_stream', (False, True))
def test_mhcv2(
    num_fracs,
    num_streams,
    disable,
    add_attn_pool_reduce_stream
):
    import torch
    from torch import nn
    # a single branch layer

    branch = nn.Linear(512, 512)

    # before

    residual = torch.randn(2, 1024, 512)

    residual = branch(residual) + residual

    # after, say 4 streams in paper

    from hyper_connections.mHCv2 import get_init_and_expand_reduce_stream_functions

    init_hyper_conn, expand_stream, reduce_stream = get_init_and_expand_reduce_stream_functions(num_streams, dim = 512, num_fracs = num_fracs, disable = disable, add_attn_pool_reduce_stream = add_attn_pool_reduce_stream)

    # 1. wrap your branch function

    hyper_conn_branch = init_hyper_conn(dim = 512, branch = branch)

    # 2. expand to 4 streams, this must be done before your trunk, typically a for-loop with many branch functions

    residual = expand_stream(residual)

    # 3. forward your residual as usual into the wrapped branch function(s)

    residual = hyper_conn_branch(residual)

    # 4. reduce 4 streams with a summation, this has to be done after your for-loop trunk. for transformer, unsure whether to do before or after final norm

    residual = reduce_stream(residual)

    assert residual.shape == (2, 1024, 512)

def test_triton_sinkhorn():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from hyper_connections.triton_sinkhorn import triton_sinkhorn
    from hyper_connections.mHCv2 import sinkhorn_knopps, log_domain_sinkhorn_knopps

    B, M, N = 2, 16, 16
    log_alpha = torch.randn(B, M, N, device = 'cuda', requires_grad = True, dtype = torch.float32)
    iters = 20

    # 1. Forward equivalence with sinkhorn_knopps
    out_triton = triton_sinkhorn(log_alpha, iters = iters)
    out_torch = sinkhorn_knopps(log_alpha, iters = iters)
    torch.testing.assert_close(out_triton, out_torch, atol = 1e-4, rtol = 1e-4)

    # 2. Forward equivalence with log_domain_sinkhorn_knopps
    out_log_torch = log_domain_sinkhorn_knopps(log_alpha, iters = iters)
    torch.testing.assert_close(out_triton, out_log_torch, atol = 1e-4, rtol = 1e-4)

    # 3. Backward parity check
    out_triton.backward(torch.ones_like(out_triton))
    grad_triton = log_alpha.grad.clone()

    log_alpha.grad.zero_()
    out_torch.backward(torch.ones_like(out_torch))
    grad_torch = log_alpha.grad.clone()

    torch.testing.assert_close(grad_triton, grad_torch, atol = 1e-3, rtol = 1e-3)

    log_alpha_double = torch.randn(1, 4, 4, device = 'cuda', requires_grad = True, dtype = torch.float64)

    torch.autograd.gradcheck(triton_sinkhorn, (log_alpha_double, 10), eps = 1e-6, atol = 1e-5)
