from pathlib import Path
import csv,numpy as np
root=Path(__file__).resolve().parents[1]
with (root/'data/splits/val.csv').open(encoding='utf-8-sig') as f: sample_id=next(csv.DictReader(f))['sample_id']
for setting in ['original','adapter_bf16']:
    with np.load(root/'results'/setting/'val'/(sample_id+'.npz'),allow_pickle=False) as data:
        print(setting,data['sample_id'].item(),data['probability'].shape,data['mask_embedding'].shape)
