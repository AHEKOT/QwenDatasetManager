import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ai_toolkit'))
import json
import random
import sqlite3
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
from toolkit.config_modules import ModelConfig
from huggingface_hub import hf_hub_download
from extensions.rgba_training.qwen_image_edit_plus_rgba import QwenImageEditPlusRGBAModel

root = Path(__file__).resolve().parents[1]
with sqlite3.connect(root / 'trainer/trainer.db') as c:
    data = c.execute('SELECT job_config FROM Job WHERE name=?', ('VNCCS_QIE2511_PoseStudiov7_Transparent',)).fetchone()
config = json.loads(data[0])['config']['process'][0]
config['model']['name_or_path'] = str(Path(hf_hub_download(config['model']['name_or_path'], 'transformer/config.json', local_files_only=True)).parent.parent)
if '--train-smoke' in sys.argv:
    import shutil
    import tempfile
    from toolkit.job import get_job
    os.environ.pop('AITK_JOB_ID', None)
    with tempfile.TemporaryDirectory(prefix='.qie_smoke_', dir=root / 'trainer') as folder:
        smoke_root = Path(folder)
        assert smoke_root.resolve().parent == (root / 'trainer').resolve()
        dataset = config['datasets'][0]
        image_root = smoke_root / 'img'
        image_root.mkdir()
        controls = dataset['control_path']
        new_controls = []
        for i in range(len(controls)):
            dest = smoke_root / f'Control{i+1}'
            dest.mkdir()
            new_controls.append(str(dest))
        for target in sorted(Path(dataset['folder_path']).glob('*.png'))[:2]:
            shutil.copy2(target, image_root / target.name)
            shutil.copy2(target.with_suffix('.txt'), image_root / target.with_suffix('.txt').name)
            for source, dest in zip(controls, new_controls):
                reference = next(p for p in Path(source).iterdir() if p.stem == target.stem and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
                shutil.copy2(reference, Path(dest) / reference.name)
        dataset.update(folder_path=str(image_root), control_path=new_controls, resolution=[512], num_workers=0)
        config['datasets'] = [dataset]
        config['training_folder'] = str(smoke_root / 'output')
        config.pop('sqlite_db_path', None)
        config['train'].update(steps=2, disable_sampling=True, skip_first_sample=True)
        config['save'].update(save_every=9999, max_step_saves_to_keep=1)
        job_config = json.loads(data[0])
        job_config['config'].update(name='qie_memory_smoke', process=[config])
        job_config.pop('meta', None)
        random.seed(42)
        job = get_job(job_config)
        job.run()
        job.cleanup()
        print('TRAINING SMOKE PASSED', flush=True)
    raise SystemExit(0)
random.seed(42)
model = QwenImageEditPlusRGBAModel('cuda:0', ModelConfig(**config['model']), dtype='bf16')

def memory(label):
    torch.cuda.synchronize()
    print(label, 'allocated GiB', round(torch.cuda.memory_allocated()/2**30, 3),
          'peak GiB', round(torch.cuda.max_memory_allocated()/2**30, 3), flush=True)

model.load_model()
memory('LOADED')
model.set_device_state_preset('cache_text_encoder')
memory('CACHE PHASE')
dataset = config['datasets'][0]
target = sorted(Path(dataset['folder_path']).glob('*.png'))[0]
caption = target.with_suffix('.txt').read_text(encoding='utf-8')
controls = []
for folder in dataset['control_path']:
    path = next(p for p in Path(folder).iterdir() if p.stem == target.stem and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        print('CONTROL', image.size, flush=True)
        controls.append(to_tensor(image).unsqueeze(0).to('cuda:0', dtype=torch.bfloat16))
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    for i in range(2):
        embeds = model.encode_prompt(caption, control_images=controls)
        print('ENCODED', i, tuple(embeds.text_embeds.shape), flush=True)
        memory('ENCODING')
        del embeds
print('PROBE PASSED', flush=True)
