import torch
from torch import nn

from zeta.structs.auto_regressive_wrapper import AutoRegressiveWrapper


def test_autoregressive_wrapper_initialization():
    net = nn.Linear(10, 10)
    wrapper = AutoRegressiveWrapper(net)

    assert isinstance(wrapper, AutoRegressiveWrapper)
    assert wrapper.net == net
    assert wrapper.max_seq_len == net.max_seq_len
    assert wrapper.pad_value == 0
    assert wrapper.ignore_index == -100
    assert wrapper.mask_prob == 0.0


def test_autoregressive_wrapper_forward():
    net = nn.Linear(10, 10)
    wrapper = AutoRegressiveWrapper(net)

    x = torch.randn(1, 10)
    logits = wrapper(x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == torch.Size([1, 10, 10])


def test_autoregressive_wrapper_generate():
    net = nn.Linear(10, 10)
    wrapper = AutoRegressiveWrapper(net)

    x = torch.randn(1, 10)
    generated = wrapper.generate(x, 10)

    assert isinstance(generated, torch.Tensor)
    assert generated.shape == torch.Size([1, 10])
