import torch
import triton
import triton.language as tl
from torch.autograd import Function

@triton.jit
def sinkhorn_kernel_forward_log(
    input_ptr,
    output_ptr,
    M, N,
    stride_b, stride_m, stride_n,
    iters: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    curr_input_ptr = input_ptr + pid_b * stride_b
    # Use a large negative value for log-space padding to avoid interference
    log_alpha = tl.load(curr_input_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask, other=-1e10)
    
    # Use static_range to force unrolling and avoid compiler bugs with dynamic loops in this environment
    for _ in tl.static_range(iters):
        # Column-wise Log-Softmax (dim=-2)
        col_max = tl.max(tl.where(mask, log_alpha, -1e10), axis=0)
        exp_weights_col = tl.exp(log_alpha - col_max[None, :])
        exp_weights_col = tl.where(mask, exp_weights_col, 0.0)
        col_lse = col_max + tl.log(tl.sum(exp_weights_col, axis=0))
        log_alpha = log_alpha - col_lse[None, :]
        log_alpha = tl.where(mask, log_alpha, -1e10)
        
        # Row-wise Log-Softmax (dim=-1)
        row_max = tl.max(tl.where(mask, log_alpha, -1e10), axis=1)
        exp_weights_row = tl.exp(log_alpha - row_max[:, None])
        exp_weights_row = tl.where(mask, exp_weights_row, 0.0)
        row_lse = row_max + tl.log(tl.sum(exp_weights_row, axis=1))
        log_alpha = log_alpha - row_lse[:, None]
        log_alpha = tl.where(mask, log_alpha, -1e10)
    
    result_alpha = tl.exp(log_alpha)
    result_alpha = tl.where(mask, result_alpha, 0.0)
    
    curr_output_ptr = output_ptr + pid_b * stride_b
    tl.store(curr_output_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, result_alpha, mask=mask)

@triton.jit
def sinkhorn_kernel_backward_log(
    grad_output_ptr,
    output_ptr,
    grad_input_ptr,
    M, N,
    stride_b, stride_m, stride_n,
    iters: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    curr_output_ptr = output_ptr + pid_b * stride_b
    curr_grad_output_ptr = grad_output_ptr + pid_b * stride_b
    
    alpha = tl.load(curr_output_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask, other=0.0)
    grad_alpha = tl.load(curr_grad_output_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask, other=0.0)
    
    # Ensure they are truly zeroed in padded areas for sum robustness
    alpha = tl.where(mask, alpha, 0.0)
    grad_alpha = tl.where(mask, grad_alpha, 0.0)
    
    for _ in tl.static_range(iters):
        # Backward of Row-wise Normalization
        # Sum only over valid elements
        row_sum_grad_alpha = tl.sum(tl.where(mask, grad_alpha * alpha, 0.0), axis=1)
        grad_alpha = grad_alpha - row_sum_grad_alpha[:, None]
        grad_alpha = tl.where(mask, grad_alpha, 0.0)
        
        # Backward of Column-wise Normalization
        col_sum_grad_alpha = tl.sum(tl.where(mask, grad_alpha * alpha, 0.0), axis=0)
        grad_alpha = grad_alpha - col_sum_grad_alpha[None, :]
        grad_alpha = tl.where(mask, grad_alpha, 0.0)
    
    grad_input = alpha * grad_alpha
    
    curr_grad_input_ptr = grad_input_ptr + pid_b * stride_b
    tl.store(curr_grad_input_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, grad_input, mask=mask)

class TritonSinkhornFunction(Function):
    @staticmethod
    def forward(ctx, log_alpha, iters=20):
        # Handle matrix size limits to avoid register spilling/SRAM overflow
        M, N = log_alpha.shape[-2:]
        if max(M, N) > 256:
             from hyper_connections.mHCv2 import log_domain_sinkhorn_knopps
             return log_domain_sinkhorn_knopps(log_alpha, iters)

        batch_shape = log_alpha.shape[:-2]
        log_alpha_flat = log_alpha.view(-1, M, N).contiguous()
        B = log_alpha_flat.shape[0]
        
        output = torch.empty_like(log_alpha_flat)
        BLOCK_SIZE = max(32, triton.next_power_of_2(max(M, N)))
        
        sinkhorn_kernel_forward_log[(B,)](
            log_alpha_flat,
            output,
            M, N,
            log_alpha_flat.stride(0), log_alpha_flat.stride(1), log_alpha_flat.stride(2),
            iters=iters,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        ctx.save_for_backward(output)
        ctx.iters = iters
        return output.view(*batch_shape, M, N)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        iters = ctx.iters
        B, M, N = output.shape
        BLOCK_SIZE = max(32, triton.next_power_of_2(max(M, N)))
        
        # Explicit contiguity for grad_output
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(output)
        
        sinkhorn_kernel_backward_log[(B,)](
            grad_output.view(B, M, N),
            output,
            grad_input,
            M, N,
            grad_input.stride(0), grad_input.stride(1), grad_input.stride(2),
            iters=iters,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        return grad_input.view_as(grad_output), None

def triton_sinkhorn(log_alpha, iters=20):
    if log_alpha.is_cuda:
        try:
            return TritonSinkhornFunction.apply(log_alpha, iters)
        except Exception:
            pass
    
    # fallback
    from hyper_connections.mHCv2 import sinkhorn_knopps
    return sinkhorn_knopps(log_alpha, iters = iters)

def is_triton_available():
    try:
        import triton
        return torch.cuda.is_available()
    except ImportError:
        return False
