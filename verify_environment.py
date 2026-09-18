"""Check actual HEIC inputs, pretrained weights, and a full GPU backward step."""
import json
import argparse
from pathlib import Path
import torch
from mambavision import create_model
from hand_data import collect
from train_mambavision import HandDataset

assert torch.cuda.is_available(), 'CUDA unavailable'
root = Path(__file__).resolve().parent
cache = Path.home() / '.cache/hand_mambavision'
cache.mkdir(parents=True, exist_ok=True)
with torch.serialization.safe_globals([argparse.Namespace]):
    model = create_model('mamba_vision_T', pretrained=True,
                         model_path=str(cache / 'mambavision_tiny_1k.pth.tar'))
model.head = torch.nn.Linear(model.head.in_features, 2)
model.cuda().train()
rows = collect(root / 'hand_img', 'train')
selected = [next(r for r in rows if r['label'] == label and r['path'].lower().endswith('.heic'))
            for label in (0, 1)]
dataset = HandDataset(root / 'hand_img', selected, 224, model.default_cfg['mean'], model.default_cfg['std'])
batch = [dataset[i] for i in range(2)]
images = torch.stack([x for x, _ in batch]).cuda()
labels = torch.tensor([y for _, y in batch], device='cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad(set_to_none=True)
logits = model(images)
assert tuple(logits.shape) == (2, 2)
loss = torch.nn.functional.cross_entropy(logits, labels)
assert torch.isfinite(loss)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
optimizer.step()
torch.cuda.synchronize()
report = {'torch': torch.__version__, 'cuda': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(0), 'input_shape': list(images.shape),
          'output_shape': list(logits.shape), 'loss': loss.item(),
          'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
          'heic_decode': 'passed', 'pretrained_load': 'passed', 'forward_backward_optimizer': 'passed'}
(root / 'environment_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
