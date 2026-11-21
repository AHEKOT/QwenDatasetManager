"""CPU-only contract checks. No checkpoints, downloads, ComfyUI server or GPU."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qdm_qi2_layer_core', ROOT / 'core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class LayoutProbe:
    def txt_in(self, x):
        return x
    def img_in(self, x):
        return x
    def pe_embedder(self, ids):
        self.ids = ids[0]
        return ids[:, None]


class LayerContractTests(unittest.TestCase):
    def layout(self, slots=3):
        probe = LayoutProbe()
        x = torch.zeros(1,64,slots*2,4)
        for i, chunk in enumerate(x.chunk(slots,dim=2)):
            chunk.fill_(i+1)
        text = torch.zeros(1,3,64)
        ref = torch.full((1,64,2,2),99.)
        states, _, segments = core.joint_build_sequence(probe,x,text,[ref],[1],layer_slots=slots)
        return probe, states, segments

    def test_joint_targets_not_independent_or_causal_layers(self):
        probe, states, segments = self.layout()
        self.assertEqual([(a,b) for a,b,_ in segments], [(0,1),(1,5),(5,7),(7,31)])
        self.assertIsNone(segments[-1][2])
        # Last segment sees every key up to its end: lower layer sees upper and vice versa.
        mask = torch.zeros(31,31,dtype=torch.bool)
        for start,end,local in segments:
            mask[start:end,:end] = True if local is None else local
        self.assertTrue(mask[7,30] and mask[30,7])
        self.assertFalse(mask[:7,7:].any())
        for i in range(3):
            self.assertTrue(torch.all(states[:,7+i*8:7+(i+1)*8] == i+1))

    def test_layer_spatial_coordinates_reset_and_order_advances(self):
        probe,_,_ = self.layout()
        grids = probe.ids[7:].reshape(3,8,3)
        torch.testing.assert_close(grids[0,:,1:],grids[2,:,1:])
        self.assertEqual(grids[:,0,0].tolist(),[5.,6.,7.])
        self.assertEqual(grids[0,:,1].tolist(),[-1.]*4+[0.]*4)
        self.assertEqual(grids[0,:,2].tolist(),[-2.,-1.,0.,1.]*2)

    def test_rope_matches_actual_training_implementation(self):
        path = ROOT.parent / 'trainer/ai_toolkit/extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py'
        if not path.is_file():
            self.skipTest('Training source is not installed alongside this standalone pack.')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'QwenImage21Rope')
        scope = {'torch':torch,'nn':torch.nn}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
        rope = scope['QwenImage21Rope'](10000,[16,56,56])
        for slots in (1,3,20):
            probe,_,_ = self.layout(slots)
            mask = torch.tensor([False]+[True]*4+[False]*2+[True]*(slots*8))
            actual = rope([(1,2,2)]+[(1,2,4)]*slots, mask, torch.device('cpu'), num_target_images=slots)
            expected = torch.cat([rope.rope_params(probe.ids[:,i],[16,56,56][i]) for i in range(3)],dim=-1)
            torch.testing.assert_close(actual,expected)

    def test_scheduler_matches_training_diffusers(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        for slots in (4,20):
            schedule = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift_terminal=.02)
            tokens=slots*64*64
            mu=.5+(tokens-256)*.4/(8192-256)
            schedule.set_timesteps(sigmas=np.linspace(1,1/40,40),mu=mu,device='cpu')
            torch.testing.assert_close(core.flow_sigmas(1024,1024,slots,40),schedule.sigmas)

    def test_rejects_mismatched_slots_and_geometry(self):
        with self.assertRaises(ValueError):
            core.joint_build_sequence(LayoutProbe(),torch.zeros(1,64,8,4),torch.zeros(1,3,64),[torch.zeros(1,64,2,2)],[],layer_slots=4)
        with self.assertRaises(ValueError):
            core.flow_sigmas(1025,1024,4,40)
        with self.assertRaises(ValueError):
            core.flow_sigmas(1024,1024,4,1)

    def test_psd_alpha_order_and_empty_slots(self):
        from psd_tools import PSDImage
        layers=[Image.new('RGBA',(16,16),(255,0,0,255)),Image.new('RGBA',(16,16),(0,0,255,128)),Image.new('RGBA',(16,16),(0,0,0,0))]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'layers.psd'
            core.save_psd(path,layers)
            psd=PSDImage.open(path)
            self.assertEqual([x.name for x in psd],['Layer 01','Layer 02','Layer 03'])
            for original,restored in zip(layers,psd):
                np.testing.assert_array_equal(np.asarray(original),np.asarray(restored.topil().convert('RGBA')))
            self.assertEqual(core.composite_layers(layers).getpixel((0,0)),(127,0,128,255))

    def test_workflow_links_and_training_slots(self):
        for slots in (4,20):
            graph=json.loads((ROOT/f'workflows/QI2_Image_to_PSD_{slots}layers.json').read_text(encoding='utf-8'))
            nodes={n['id']:n for n in graph['nodes']}
            self.assertEqual(nodes[5]['widgets_values'][1],slots)
            for link,src,out,dst,inp,typ in graph['links']:
                self.assertEqual(nodes[src]['outputs'][out]['type'],typ)
                self.assertEqual(nodes[dst]['inputs'][inp]['type'],typ)
                self.assertIn(link,nodes[src]['outputs'][out]['links'])
                self.assertEqual(nodes[dst]['inputs'][inp]['link'],link)
            api=json.loads((ROOT/f'workflows/QI2_Image_to_PSD_{slots}layers_api.json').read_text(encoding='utf-8'))
            self.assertEqual(api['5']['inputs']['layer_slots'],slots)
            self.assertEqual(api['6']['inputs']['prompt'],'Create layered image from image 1')


if __name__ == '__main__':
    unittest.main()
