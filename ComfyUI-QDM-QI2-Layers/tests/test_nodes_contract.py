"""Node boundary checks with Comfy transports stubbed, no ComfyUI runtime."""
import ast
import functools
import inspect
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
import uuid
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn.functional as F

from test_core import core, ROOT


def node_scope():
    tree=ast.parse((ROOT/'nodes.py').read_text(encoding='utf-8'))
    # Comfy imports are replaced by controlled transports; execute actual node functions/classes.
    body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    scope={'__name__':'qdm_nodes_contract','functools':functools,'inspect':inspect,'json':json,
           'math':math,'Path':Path,'types':types,'uuid':uuid,'np':np,'torch':torch,'F':F,
           **{k:getattr(core,k) for k in ('FORMAT_VERSION','validate_geometry','joint_build_sequence','flow_sigmas','to_pil_layers','composite_layers','save_psd','legacy_timestep_forward')}}
    exec(compile(ast.Module(body=body,type_ignores=[]),'nodes.py','exec'),scope)
    return scope


class NodeBoundaryTests(unittest.TestCase):
    def test_metadata_rejects_ordinary_adapter_and_reads_slots(self):
        scope=node_scope()
        parse=scope['declared_slots']
        self.assertEqual(parse({'qdm':json.dumps({'trainingPreset':'qwen_layered_lora','layerFormat':{'layerSlots':20}})}),20)
        self.assertIsNone(parse({}))
        with self.assertRaises(ValueError):
            parse({'qdm':json.dumps({'trainingPreset':'standard_lora'})})

    def test_timestep_contract_supports_old_and_corrected_adapters(self):
        parse=node_scope()['declared_timestep_embedding']
        self.assertEqual(parse({'qdm':json.dumps({'layerFormat':{'version':1}})}),'legacy_bf16')
        self.assertEqual(parse({'qdm':json.dumps({'layerFormat':{'timestepEmbedding':'fp32'}})}),'fp32')
        self.assertEqual(parse({'meta':json.dumps({'qdm':{'layerFormat':{'timestepEmbedding':'legacy_bf16'}}})}),'legacy_bf16')
        with self.assertRaises(ValueError):
            parse({'qdm':{'layerFormat':{'timestepEmbedding':'unknown'}}})

    def test_decode_never_passes_stacked_height_to_vae(self):
        scope=node_scope()
        scope['mm']=types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        inputs=[]
        def decode(value):
            inputs.append(tuple(value.shape))
            pixel=torch.zeros(1,32,64,4)
            pixel[...,0]=value[0,0,0,0]
            pixel[...,3]=1 if len(inputs)==1 else .5
            return pixel
        vae=types.SimpleNamespace(latent_channels=64,output_channels=4,downscale_ratio=16,decode=decode)
        latent=torch.cat([torch.full((1,64,2,4),v) for v in (.2,.8)],dim=2)
        meta={'layer_slots':2,'width':64,'height':32}
        doc,composite,rgba,masks=scope['QDMLayeredQI2Decode']().decode({'samples':latent,'metadata':meta},vae)
        self.assertEqual(inputs,[(1,64,2,4)]*2)
        self.assertEqual(tuple(rgba.shape),(2,32,64,4))
        torch.testing.assert_close(masks,1-rgba[...,3])
        self.assertEqual(doc['layers'][0].getpixel((0,0))[0],51)
        self.assertEqual(doc['layers'][1].getpixel((0,0))[0],204)

    def test_sampler_patches_only_clone_and_shares_layer_count(self):
        scope=node_scope()
        native=types.SimpleNamespace(time_text_embed=object())
        scope['require_model']=lambda model:native
        config={'format_version':1,'layer_slots':4,'lora_name':'test.safetensors','strength':1.0,'timestep_embedding':'fp32'}
        clone=types.SimpleNamespace(model_options={'qdm_qi2_layers':dict(config),'transformer_options':{}},
                                    add_object_patch=Mock(),load_device='cpu')
        original=types.SimpleNamespace(model_options={'qdm_qi2_layers':dict(config)},clone=lambda:clone)
        calls=[]
        def sample(*args,**kwargs):
            calls.append((args,kwargs))
            return args[7]
        scope['comfy']=types.SimpleNamespace(sample=types.SimpleNamespace(sample_custom=sample),
            samplers=types.SimpleNamespace(sampler_object=lambda name:name),
            utils=types.SimpleNamespace(ProgressBar=lambda steps:Mock(),PROGRESS_BAR_ENABLED=False))
        scope['mm']=types.SimpleNamespace(intermediate_device=lambda:torch.device('cpu'))
        cond={'width':64,'height':32,'positive':[],'negative':[],'prompt':'test','negative_prompt':'',
              'has_reference':True,'match_reference_area':True}
        result=scope['QDMLayeredQI2Sampler']().sample(original,cond,123,10,1.)[0]
        self.assertEqual(tuple(result['samples'].shape),(1,64,8,4))
        self.assertNotIn('transformer_options',original.model_options)
        clone.add_object_patch.assert_called_once()
        self.assertEqual(clone.add_object_patch.call_args.args[0],'diffusion_model.build_sequence')
        self.assertIs(clone.add_object_patch.call_args.args[1].__self__,native)
        self.assertEqual(calls[0][0][3],'euler')
        torch.testing.assert_close(calls[0][0][4],core.flow_sigmas(64,32,4,10))
        original.model_options['qdm_qi2_layers']['timestep_embedding']='legacy_bf16'
        clone.add_object_patch.reset_mock()
        scope['QDMLayeredQI2Sampler']().sample(original,cond,123,10,1.)
        self.assertEqual(clone.add_object_patch.call_args_list[-1].args[0],'diffusion_model.time_text_embed.forward')
        self.assertIs(clone.add_object_patch.call_args_list[-1].args[1].__self__,native.time_text_embed)

    def test_save_preserves_order_and_reports_actual_path(self):
        from PIL import Image
        scope=node_scope()
        with tempfile.TemporaryDirectory() as tmp:
            scope['folder_paths']=types.SimpleNamespace(get_output_directory=lambda:tmp,
                get_save_image_path=lambda prefix,root,w,h:(str(Path(tmp)/'QI2-Layers'),'layered',1,'QI2-Layers',prefix))
            layers=[Image.new('RGBA',(16,16),(255,0,0,255)),Image.new('RGBA',(16,16),(0,255,0,128))]
            out=scope['QDMLayeredQI2Save']().save({'layers':layers,'metadata':{'layer_slots':2}},'QI2-Layers/layered')
            path=Path(out['result'][0])
            self.assertTrue(path.is_file())
            self.assertTrue((path.parent/'layers/01.png').is_file())
            np.testing.assert_array_equal(np.asarray(Image.open(path.parent/'layers/02.png')),np.asarray(layers[1]))
            meta=json.loads((path.parent/'metadata.json').read_text())
            self.assertEqual(meta['order'],'bottom_to_top')
            preview=out['ui']['images'][0]
            self.assertTrue((Path(tmp)/preview['subfolder']/preview['filename']).is_file())


if __name__=='__main__':
    unittest.main()
