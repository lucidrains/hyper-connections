<img src="./hyper-connections.png" width="450px"></img>

## Hyper Connections

Attempt to make multiple residual streams, proposed in [Hyper-Connections paper](https://arxiv.org/abs/2409.19606) out of Bytedance AI lab, accessible as an easy to use library, as well as for following any new research in this direction.

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

from hyper_connections import HyperConnections

init_hyper_conn, expand_stream, reduce_stream = HyperConnections.get_init_and_expand_reduce_stream_functions(4)

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

from hyper_connections import HyperConnections

init_hyper_conn, expand_stream, reduce_stream = HyperConnections.get_init_and_expand_reduce_stream_functions(4)

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
HyperConnections.get_init_and_expand_reduce_stream_functions(4, disable = True)
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
