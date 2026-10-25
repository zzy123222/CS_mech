import torch.nn as nn
from src.utils.module_access import ModuleSpec, get_decoder_layers, resolve_module


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()


def test_module_spec_parse_and_resolve():
    m = Model()
    assert len(get_decoder_layers(m)) == 2
    assert ModuleSpec.parse("L1:mlp") == ModuleSpec(1, "mlp")
    assert resolve_module(m, ModuleSpec(0, "attn")) is m.model.layers[0].self_attn
