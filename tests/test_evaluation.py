import torch
from t2m.model import PrefixConditionedLM
from t2m.eval import evaluate_text
from t2m.data import collate
from test_model import DummyCausalLM


class FakeDataset:
    def __len__(self): return 20
    def __getitem__(self,idx):
        z=torch.zeros(768);z[idx%20]=1
        return z,torch.tensor([3+(idx%10),4,2])


def test_evaluation_metrics():
    model=PrefixConditionedLM(DummyCausalLM(),adapter_hidden=32)
    metrics=evaluate_text(model,FakeDataset(),bos_id=1,pad_id=0,
                          device=torch.device('cpu'),batch_size=4,n_batches=2)
    for k in ('delta_cond','rank_acc','delta_zero','early_delta_cond','late_delta_cond'):
        assert k in metrics
