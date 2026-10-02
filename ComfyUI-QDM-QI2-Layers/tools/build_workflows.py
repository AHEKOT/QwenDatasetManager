"""Generate standard ComfyUI UI and API workflows; does not run ComfyUI."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1] / 'workflows'
ROOT.mkdir(exist_ok=True)


def build(slots):
    specs = [
        (1, 'UNETLoader', 'QI2 2.1 diffusion model', [0, 0], [360, 90], [], [('MODEL','MODEL')], ['qwen_image_2.1_int8_convrot.safetensors','default']),
        (2, 'CLIPLoader', 'Qwen3-VL 8B / qwen_image', [0, 140], [360, 110], [], [('CLIP','CLIP')], ['qwen3vl_8b_bf16.safetensors','qwen_image','default']),
        (3, 'VAELoader', 'Native QI2 RGBA VAE', [0, 310], [360, 65], [], [('VAE','VAE')], ['qwen_image_2.1_vae_bf16.safetensors']),
        (4, 'LoadImage', 'Flat input image (Control1)', [0, 440], [360, 340], [], [('IMAGE','IMAGE'),('MASK','MASK')], ['select_input_image.png','image']),
        (5, 'QDMLayeredQI2LoRA', f'Select your trained LoRA / {slots} slots', [440, 0], [400, 160], [('model','MODEL')], [('MODEL','MODEL')], ['select_your_trained_layer_lora.safetensors',slots,1.0]),
        (6, 'QDMLayeredQI2Conditioning', 'Instruction + reference', [440, 260], [420, 320], [('clip','CLIP'),('vae','VAE'),('image','IMAGE')], [('CONDITIONING','QDM_QI2_CONDITIONING')], ['Create layered image from image 1','',1024,1024,True]),
        (7, 'QDMLayeredQI2Sampler', 'Joint denoising of every layer', [930, 0], [350, 170], [('model','MODEL'),('conditioning','QDM_QI2_CONDITIONING')], [('LATENT','QDM_QI2_LAYER_LATENT')], [0,'fixed',40,1.0]),
        (8, 'QDMLayeredQI2Decode', 'Decode each RGBA layer', [930, 260], [350, 140], [('samples','QDM_QI2_LAYER_LATENT'),('vae','VAE')], [('document','QDM_RGBA_LAYERS'),('composite_rgba','IMAGE'),('layers_rgba','IMAGE'),('transparency_masks','MASK')], []),
        (9, 'QDMLayeredQI2Save', 'Save PSD + composite + RGBA layers', [1360, 0], [430, 350], [('document','QDM_RGBA_LAYERS')], [('psd_path','STRING')], ['QI2-Layers/layered']),
        (10, 'PreviewImage', 'All layers / bottom to top', [1360, 410], [430, 390], [('images','IMAGE')], [], []),
        (11, 'Note', 'Read before queueing', [440, 650], [820, 230], [], [], [
            f'QDM QI2 JOINT LAYERS — {slots} slots\n'
            '1. Install ComfyUI-QDM-QI2-Layers and its requirements in ComfyUI.\n'
            '2. Select native Qwen Image 2.1, Qwen3-VL 8B (qwen_image), and the QI2 RGBA VAE.\n'
            f'3. Select a trained QDM joint-layer LoRA with exactly {slots} training slots.\n'
            '4. Load one flat image; adjust output width/height if needed. Then queue.\n'
            'PSD + PNG + individual RGBA layers are saved under output/QI2-Layers/.\n'
            'This is one document per run. More slots use much more VRAM.\n'
            'Ordinary Qwen-Image-Layered / Qwen Image 1.x adapters are incompatible.'
        ]),
    ]
    edges = [(1,0,5,0,'MODEL'),(2,0,6,0,'CLIP'),(3,0,6,1,'VAE'),(4,0,6,2,'IMAGE'),
             (5,0,7,0,'MODEL'),(6,0,7,1,'QDM_QI2_CONDITIONING'),(7,0,8,0,'QDM_QI2_LAYER_LATENT'),
             (3,0,8,1,'VAE'),(8,0,9,0,'QDM_RGBA_LAYERS'),(8,2,10,0,'IMAGE')]
    nodes = []
    for node_id, typ, title, pos, size, inputs, outputs, widgets in specs:
        node = {'id':node_id,'type':typ,'pos':pos,'size':size,'flags':{},'order':node_id-1,'mode':0,
                'inputs':[{'name':name,'type':kind,'link':None} for name,kind in inputs],
                'outputs':[{'name':name,'type':kind,'links':[],'slot_index':i} for i,(name,kind) in enumerate(outputs)],
                'properties':{'Node name for S&R':typ},'widgets_values':widgets,'title':title}
        nodes.append(node)
    by_id = {n['id']:n for n in nodes}
    links=[]
    for link_id,(src,src_slot,dst,dst_slot,kind) in enumerate(edges,1):
        by_id[src]['outputs'][src_slot]['links'].append(link_id)
        by_id[dst]['inputs'][dst_slot]['link']=link_id
        links.append([link_id,src,src_slot,dst,dst_slot,kind])
    graph = {'last_node_id':11,'last_link_id':len(links),'nodes':nodes,'links':links,'groups':[],
             'config':{},'extra':{'ds':{'scale':.8,'offset':[40,40]}},'version':.4}
    api = {
        '1': {'class_type':'UNETLoader','inputs':{'unet_name':specs[0][-1][0],'weight_dtype':'default'}},
        '2': {'class_type':'CLIPLoader','inputs':{'clip_name':specs[1][-1][0],'type':'qwen_image','device':'default'}},
        '3': {'class_type':'VAELoader','inputs':{'vae_name':specs[2][-1][0]}},
        '4': {'class_type':'LoadImage','inputs':{'image':'select_input_image.png'}},
        '5': {'class_type':'QDMLayeredQI2LoRA','inputs':{'model':['1',0],'lora_name':specs[4][-1][0],'layer_slots':slots,'strength':1.0}},
        '6': {'class_type':'QDMLayeredQI2Conditioning','inputs':{'clip':['2',0],'vae':['3',0],'image':['4',0],
             'prompt':'Create layered image from image 1','negative_prompt':'','width':1024,'height':1024,'match_reference_area':True}},
        '7': {'class_type':'QDMLayeredQI2Sampler','inputs':{'model':['5',0],'conditioning':['6',0],'seed':0,'steps':40,'cfg':1.0}},
        '8': {'class_type':'QDMLayeredQI2Decode','inputs':{'samples':['7',0],'vae':['3',0]}},
        '9': {'class_type':'QDMLayeredQI2Save','inputs':{'document':['8',0],'filename_prefix':'QI2-Layers/layered'}},
        '10': {'class_type':'PreviewImage','inputs':{'images':['8',2]}},
    }
    for suffix,data in [('',graph),('_api',api)]:
        (ROOT / f'QI2_Image_to_PSD_{slots}layers{suffix}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    build(4)
    build(20)
