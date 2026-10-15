from types import SimpleNamespace
import torch
from torch import nn
from t2m.model import PrefixConditionedLM
from t2m.data import collate, different_indices


class DummyCausalLM(nn.Module):
    """No HuggingFace downloads: verifies prefix alignment and gradient connectivity."""
    def __init__(self, vocab=47, hidden=24):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.linear = nn.Linear(hidden, vocab, bias=False)
    def get_input_embeddings(self):
        return self.embed
    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=False, **kwargs):
        # Position t depends on prefix + tokens 0..t (causal running mean).
        valid = attention_mask.unsqueeze(-1)
        x = inputs_embeds * valid
        h = x.cumsum(1) / valid.cumsum(1).clamp(min=1)
        return SimpleNamespace(logits=self.linear(h),past_key_values=None)


def make_batch():
    rows = [(torch.randn(768),torch.tensor([3,4,5,2])),
            (torch.randn(768),torch.tensor([6,7,8,2]))]
    return collate(rows,bos_id=1,pad_id=0)


def test_labels_align_prefix_and_bos():
    m = PrefixConditionedLM(DummyCausalLM(), adapter_hidden=32)
    batch = make_batch()
    logits, token_loss, counts = m.token_logprobs(batch['embeddings'],batch['input_ids'],
                                                  batch['labels'],batch['mask'])
    assert logits.shape == (2,)
    assert token_loss.shape == (2,4)
    assert counts.tolist() == [4,4]


def test_rank_backprop_through_prefix_and_frozen_lm():
    m = PrefixConditionedLM(DummyCausalLM(), adapter_hidden=32)
    m.lm.requires_grad_(False)
    b = make_batch()
    loss, diagnostics = m(b['embeddings'], b['input_ids'], b['labels'], b['mask'],
                          torch.tensor([1,0]), .2,.2,.005)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.adapter.parameters())
    assert all(p.grad is None for p in m.lm.parameters())
    assert torch.isfinite(loss)
    assert set(['delta_cond','rank_acc','ce','rank_loss']).issubset(diagnostics)


def test_derangement():
    for n in range(2,13):
        for _ in range(10):
            idx = different_indices(n, torch.device('cpu'))
            assert torch.equal(idx.sort().values,torch.arange(n))
            assert torch.all(idx != torch.arange(n))
