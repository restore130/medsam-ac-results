"""Validate the delivered four split outputs and all checksummed files."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = read_csv(ROOT / 'SHA256SUMS.csv')
    for row in manifest:
        path = ROOT / row['path']
        assert path.stat().st_size == int(row['bytes']), row['path']
        assert sha256(path) == row['sha256'], row['path']
    findings = []
    for setting, folder in [('original', 'original'), ('adapter', 'adapter_bf16')]:
        for split in ['val', 'test']:
            directory = ROOT / 'results' / folder / split
            expected = read_csv(ROOT / 'data/splits' / f'{split}.csv')
            metrics = read_csv(directory / 'metrics.csv')
            assert [r['sample_id'] for r in metrics] == [r['sample_id'] for r in expected]
            assert len(expected) == len(metrics) == 150
            assert {p.stem for p in directory.glob('*.npz')} == {r['sample_id'] for r in expected}
            values = []
            for row, metric in zip(expected, metrics):
                with Image.open(ROOT / row['mask_path']) as image:
                    gt = np.asarray(image.convert('L')) > 127
                with np.load(directory / (row['sample_id'] + '.npz'), allow_pickle=False) as data:
                    assert set(data.files) == {'sample_id','probability','prediction_mask','mask_embedding','bbox_original','bbox_1024','original_size'}
                    assert data['sample_id'].item() == row['sample_id']
                    probability, prediction, embedding = data['probability'], data['prediction_mask'], data['mask_embedding']
                    assert probability.dtype == np.float32 and prediction.dtype == np.uint8
                    assert probability.shape == prediction.shape == gt.shape
                    assert embedding.dtype == np.float32 and embedding.shape == (32,256,256)
                    assert np.isfinite(probability).all() and np.isfinite(embedding).all()
                    assert probability.min() >= 0 and probability.max() <= 1
                    assert np.array_equal(prediction, probability > 0.5)
                    assert np.array_equal(data['original_size'], np.array(gt.shape))
                    ys, xs = np.where(gt)
                    box = np.array([xs.min(),ys.min(),xs.max(),ys.max()], dtype=np.float32)
                    assert np.array_equal(data['bbox_original'],box)
                    assert np.allclose(data['bbox_1024'], box / np.array([gt.shape[1],gt.shape[0]]*2,dtype=np.float32)*1024,atol=1e-4,rtol=0)
                    intersection = np.count_nonzero(prediction.astype(bool) & gt)
                    denominator = np.count_nonzero(prediction) + np.count_nonzero(gt)
                    dice, iou = 2*intersection/denominator, intersection/(denominator-intersection)
                    assert abs(dice-float(metric['dice'])) < 1e-12
                    assert abs(iou-float(metric['iou'])) < 1e-12
                    assert np.isclose(np.linalg.norm(embedding),float(metric['embedding_l2_norm']),rtol=1e-6)
                    values.append([dice,iou])
            summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            scores=np.asarray(values)
            assert abs(scores[:,0].mean()-summary['dice_mean']) < 1e-12
            assert abs(scores[:,1].mean()-summary['iou_mean']) < 1e-12
            if setting == 'adapter':
                assert summary['adapter_checkpoint_sha256'] == sha256(ROOT/'checkpoints/adapter_bf16/best.pt')
                assert summary['trained_epoch'] == 11
            findings.append({'setting':setting,'split':split,'sample_count':len(values),'dice':float(scores[:,0].mean()),'iou':float(scores[:,1].mean())})
            print(f'{setting}/{split}: 150 arrays and metrics passed',flush=True)
    report={'status':'passed','checksummed_files':len(manifest),'verified_npz_files':600,'results':findings}
    (ROOT/'verification/package_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
