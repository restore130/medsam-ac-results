"""Uniform, read-only result interface. Run from any working directory."""
from pathlib import Path
import csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load_result(model_setting, split, sample_id):
    with (ROOT/'indices/all_samples.csv').open(encoding='utf-8-sig',newline='') as f:
        row=next((r for r in csv.DictReader(f) if (r['model_setting'],r['split'],r['sample_id'])==(model_setting,split,sample_id)),None)
    if row is None: raise KeyError((model_setting,split,sample_id))
    with np.load(ROOT/row['npz_path'],allow_pickle=False) as z:
        result={k:z[k].copy() for k in z.files}
        result['prediction_mask']=z[row['mask_key']].copy()
        result['mask_embedding']=z['mask_embedding'].astype(np.float32)
    result.update(model_setting=model_setting,split=split,source_embedding_dtype=row['embedding_storage'],gt_path=str(ROOT/row['gt_path']))
    return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('model_setting',choices=['original','linear_probe','adapter','lora_r8','lora_r16']);p.add_argument('split',choices=['val','test']);p.add_argument('sample_id');a=p.parse_args()
    r=load_result(a.model_setting,a.split,a.sample_id)
    print({k:(v.shape,str(v.dtype)) for k,v in r.items() if isinstance(v,np.ndarray)})
