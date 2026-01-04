<img src="./hyper-connections.png" width="450px"></img>

## Hyper Connections

Attempt to make multiple residual streams, proposed in [Hyper-Connections paper](https://arxiv.org/abs/2409.19606) out of Bytedance AI lab, accessible as an easy to use library, as well as for following any new research in this direction.

[Write up on mHC from Subhadip Mitra](https://subhadipmitra.com/blog/2026/deepseek-mhc-manifold-constrained-hyper-connections/)

## Install

```bash
$ pip install hyper-connections
```

## Usage

```python
import torch
from torch import nn

# a single branch layer

branch = nn.Linear(512, 512)

# before

residual = torch.randn(2, 1024, 512)

residual = branch(residual) + residual

# after, say 4 streams in paper

from hyper_connections import get_init_and_expand_reduce_stream_functions

init_hyper_conn, expand_stream, reduce_stream = get_init_and_expand_reduce_stream_functions(4)

# 1. wrap your branch function

hyper_conn_branch = init_hyper_conn(dim = 512, branch = branch)

# 2. expand to 4 streams, this must be done before your trunk, typically a for-loop with many branch functions

residual = expand_stream(residual)

# 3. forward your residual as usual into the wrapped branch function(s)

residual = hyper_conn_branch(residual) 

# 4. reduce 4 streams with a summation, this has to be done after your for-loop trunk. for transformer, unsure whether to do before or after final norm

residual = reduce_stream(residual)
```

Or doing it manually, as in the paper

```python
import torch
from torch import nn

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
```

To compare hyper connections to plain residual without changing the code, just pass `disable = True` when fetching the functions

```python
get_init_and_expand_reduce_stream_functions(4, disable = True)
```

To use the fractionated feature dimensions proposed in [a follow up paper](https://arxiv.org/abs/2503.14125) by same authors, just instantiate with `num_fracs` greater than `1` as so

```python
get_init_and_expand_reduce_stream_functions(1, num_fracs = 4) # also allows you to mix streams and fractions of feature dimension
```

## Citation

```bibtex
@article{Zhu2024HyperConnections,
    title   = {Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Zihao Huang and Yutao Zeng and Yunyao Mao and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2409.19606},
    url     = {https://api.semanticscholar.org/CorpusID:272987528}
}
```

```bibtex
@misc{Rubin2024,
    author  = {Ohad Rubin},
    url     = {https://medium.com/@ohadrubin/exploring-weight-decay-in-layer-normalization-challenges-and-a-reparameterization-solution-ad4d12c24950}
}
```

```bibtex
@article{Zhu2025FracConnectionsFE,
    title   = {Frac-Connections: Fractional Extension of Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Jundong Zhou and Zihao Huang and Yutao Zeng and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2025},
    volume  = {abs/2503.14125},
    url     = {https://api.semanticscholar.org/CorpusID:277104144}
}
```

```bibtex
@misc{xie2025mhcmanifoldconstrainedhyperconnections,
    title   = {mHC: Manifold-Constrained Hyper-Connections}, 
    author  = {Zhenda Xie and Yixuan Wei and Huanqi Cao and Chenggang Zhao and Chengqi Deng and Jiashi Li and Damai Dai and Huazuo Gao and Jiang Chang and Liang Zhao and Shangyan Zhou and Zhean Xu and Zhengyan Zhang and Wangding Zeng and Shengding Hu and Yuqing Wang and Jingyang Yuan and Lean Wang and Wenfeng Liang},
    year    = {2025},
    eprint  = {2512.24880},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2512.24880}, 
}
```
