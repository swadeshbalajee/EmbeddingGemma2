import json
from t2m.mix import build
from t2m.config import load_config


def test_variant_configuration():
    b=load_config('configs/v1b.yaml')
    c=load_config('configs/v1c.yaml')
    assert b['variant']=='v1b' and c['variant']=='v1c'
    assert b['noise_std']==0 and c['noise_std']>0
    assert b['decoder_id']==c['decoder_id']


def test_local_only_mixture(tmp_path):
    sources={}
    for source in ('image','audio','video','prose'):
        p=tmp_path/f'{source}.jsonl'
        with p.open('w') as f:
            for i in range(75):
                f.write(json.dumps({'text':f'A lengthy and informative original {source} caption describing example number {i} with a distinct object.'})+'\n')
        sources[source]=str(p)
    result=build(tmp_path/'mixed',train_size=40,val_size=4,local_files=sources)
    assert result['train']==40 and result['val']==4
    assert sum(1 for _ in (tmp_path/'mixed/train.jsonl').open())==40
