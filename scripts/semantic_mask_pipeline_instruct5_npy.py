import os
import io
import tarfile
import torch
import numpy as np
from PIL import Image
import argparse
from datasets import load_dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, CLIPSegForImageSegmentation,
    OneFormerProcessor, OneFormerForUniversalSegmentation,
    BlipProcessor, BlipForConditionalGeneration
)
from clip import clip_classification
from clipseg import clipseg_segmentation
from oneformer import oneformer_coco_segmentation, oneformer_ade20k_segmentation
from blip import open_vocabulary_classification_blip
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.transforms import InterpolationMode
import re
import tqdm
import webdataset as wds
from torch.distributed import all_gather_object
import json
def normalize_class_name(name):
    name = name.lower()
    name = re.sub(r'^(a|an|the) ', '', name)  
    return name.strip()

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12306'
#export HF_ENDPOINT="https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'

input_dir = '/mnt/33t/cy/blip3o_dataset'
output_dir = '/mnt/33t/cy/mask_dataset'
base_dir = '/mnt/33t/cy/mllm_models/semantic_sam'

debug = True
if debug:
    debug_folder = "/mnt/33t/cy/mask_debug_rgb"
    os.makedirs(debug_folder, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes')
    args = parser.parse_args()
    return args

def sync_classnames_across_processes(local_classnames: set):
    """将每个进程上的 set 聚合到一起，返回全局的 union set"""
    gathered_sets = [set() for _ in range(torch.distributed.get_world_size())]
    all_gather_object(gathered_sets, local_classnames)
    global_classnames = set().union(*gathered_sets)
    return global_classnames



def semantic_infer_one_image(img: Image.Image, key, processors, models, rank):
    """对一张图像执行语义标注, 返回 shape=(256/w, w) 的单通道 numpy array, 值为 0~num_class"""
    from mmcv import imcrop
    import pycocotools.mask as maskUtils
    img_arr = np.array(img)
    anns = {'annotations': processors['sam_generator'].generate(img_arr)}
    class_ids_from_oneformer_coco = oneformer_coco_segmentation(img,
                                                                 processors['oneformer_coco'], models['oneformer_coco'], rank)
    class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(img,
                                                                     processors['oneformer_ade20k'], models['oneformer_ade20k'], rank)
    mask_class = []
    local_classnames = set()
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        coco_ids = class_ids_from_oneformer_coco[valid_mask]
        ade_ids = class_ids_from_oneformer_ade20k[valid_mask]
        coco_labels = [CONFIG_COCO_ID2LABEL['refined_id2label'].get(str(i.item()), '') for i in torch.bincount(coco_ids).topk(1).indices]
        ade_labels = [CONFIG_ADE20K_ID2LABEL['id2label'].get(str(i.item()), '') for i in torch.bincount(ade_ids).topk(1).indices]
        labels = list(set(coco_labels + ade_labels))

        x0, y0, w, h = ann['bbox']
        patch_small = imcrop(img_arr, np.array([x0, y0, x0+w, y0+h]), scale=1.2)
        patch_large = imcrop(img_arr, np.array([x0, y0, x0+w, y0+h]), scale=1.6)
        valid_mask_huge_crop = imcrop(valid_mask.numpy(), np.array([x0, y0, x0+w, y0+h]), scale=1.6)
        open_vocab_labels = open_vocabulary_classification_blip(patch_large, processors['blip'], models['blip'], rank)
        candidate_labels = list(set(labels + open_vocab_labels))
        top_labels = clip_classification(patch_small, candidate_labels, min(3, len(candidate_labels)),
                                         processors['clip'], models['clip'], rank)
        seg = clipseg_segmentation(patch_large, top_labels, processors['clipseg'], models['clipseg'], rank).argmax(0)

        valid_mask_huge_crop = torch.tensor(valid_mask_huge_crop)
        if valid_mask_huge_crop.shape != seg.shape:
            valid_mask_huge_crop = F.interpolate(valid_mask_huge_crop.unsqueeze(0).unsqueeze(0).float(), size=seg.shape, mode='nearest').squeeze(0).squeeze(0).bool()
        seg = seg.cpu().numpy()
        class_name = top_labels[torch.bincount(torch.tensor(seg[valid_mask_huge_crop.numpy()].flatten())).topk(1).indices.item()]
        class_name = normalize_class_name(class_name)
        mask_class.append((valid_mask.numpy(), class_name))
        local_classnames.add(class_name)
    
    processors['global_classnames'] = processors['global_classnames'].union(sync_classnames_across_processes(local_classnames))
    for class_name in processors['global_classnames']:
        if class_name in processors['label2index']:
            continue
        else:
            processors['label2index'][class_name] = len(processors['label2index'])
    
    semantic_mask = np.zeros((img.height, img.width), dtype=np.int32)
    for mask, name in mask_class:
        semantic_mask[mask] = processors['label2index'][name]
    
    h, w = semantic_mask.shape
    aspect = h / w
    new_h = int((256 * aspect)**0.5)
    new_w = 256 // new_h
    assert new_h * new_w == 256
    mask_img = Image.fromarray(semantic_mask).resize((new_w, new_h), resample=Image.NEAREST) 
    mask_arr = np.array(mask_img)

    debug = True
    if debug:
        def validate_resize_recovery(original_mask, resized_mask):
            recovered = Image.fromarray(resized_mask).resize(original_mask.shape[::-1], resample=Image.NEAREST)
            recovered_mask = np.array(recovered)
            error = (original_mask != recovered_mask).sum()
            total = original_mask.size
            print(f"Mismatch pixels: {error}/{total} ({100*error/total:.2f}%)")
        validate_resize_recovery(semantic_mask, mask_arr)
       
        def save_debug_images(debug_dir, key, image, mask_arr, semantic_mask):
            os.makedirs(debug_dir, exist_ok=True)
            orig_mask_img = Image.fromarray(semantic_mask.astype(np.uint16), mode='I;16')
            resized_mask_img = Image.fromarray(mask_arr.astype(np.uint16), mode='I;16')
            recovered_mask_img = resized_mask_img.resize(semantic_mask.shape[::-1], resample=Image.NEAREST)

            image.save(os.path.join(debug_dir, f"{key}_image.jpg"))
            orig_mask_img.save(os.path.join(debug_dir, f"{key}_mask_orig.png"))
            resized_mask_img.save(os.path.join(debug_dir, f"{key}_mask_resized.png"))
            recovered_mask_img.save(os.path.join(debug_dir, f"{key}_mask_recovered.png"))
        save_debug_images(debug_folder, key, img, mask_arr, semantic_mask)
    
    return mask_arr , semantic_mask 

def main(rank, args):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device() 
    processors = {
        'clip': CLIPProcessor.from_pretrained(f"{base_dir}/clip-vit-large-patch14"),
        'clipseg': AutoProcessor.from_pretrained(f"{base_dir}/clipseg-rd64-refined"),
        'oneformer_ade20k': OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large"),
        'oneformer_coco': OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large"),
        'blip': BlipProcessor.from_pretrained(f"{base_dir}/blip-image-captioning-large"),
        #'local_classnames': set(),
        'global_classnames': set(),
        'label2index': {}
    }
    processors['clipseg'].image_processor.do_resize = False
    models = {
        'clip': CLIPModel.from_pretrained(f"{base_dir}/clip-vit-large-patch14").to(device),
        'clipseg': CLIPSegForImageSegmentation.from_pretrained(f"{base_dir}/clipseg-rd64-refined").to(device),
        'oneformer_ade20k': OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to(device),
        'oneformer_coco': OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device),
        'blip': BlipForConditionalGeneration.from_pretrained(f"{base_dir}/blip-image-captioning-large").to(device),
    }

    sam = sam_model_registry["vit_h"](checkpoint=f"{base_dir}/SAM-vit-h/sam_vit_h_4b8939.pth").to(device)
    processors['sam_generator'] = SamAutomaticMaskGenerator(model=sam, points_per_side=32,
        pred_iou_thresh=0.86, stability_score_thresh=0.92, crop_n_layers=0,
        crop_n_points_downscale_factor=2, min_mask_region_area=100, output_mode='coco_rle')
    
    tar_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tar')])
    tar_files = tar_files[:7]
    local_files = tar_files[rank::args.world_size]
    
    processed_tars,processed_imgs = [], []
    resume_path = os.path.join(output_dir, f"rank{rank}_resume.json")
    if os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            resume_data = json.load(f)
            processed_tars = resume_data.get("processed_tars", [])
            processed_imgs = resume_data.get("processed_imgs", [])
            processors["label2index"] = resume_data.get("label2index", {})
            processors["global_classnames"] = set(resume_data.get("global_classnames", []))
            #processors["local_classnames"] = set(processors["global_classnames"])

    for tar_path in (local_files if rank != 0 else tqdm.tqdm(local_files, desc='Local_file_Num: ')):
        tar_name = os.path.splitext(os.path.basename(tar_path))[0]
        if tar_name in processed_tars:
            continue
        dataset = load_dataset("webdataset", data_files=tar_path, split="train")
        
        output_tar_path = os.path.join(output_dir, f"{tar_name}.tar")
        with wds.TarWriter(output_tar_path) as sink:
            for idx, item in enumerate(dataset if rank != 0 else tqdm.tqdm(dataset, desc='DatasetLens: ')):
                if idx > 20:
                    break
                key = item["__key__"]
                if key in processed_imgs:
                    continue
                img: Image.Image = item["jpg"]
                mask_arr, semantic_mask = semantic_infer_one_image(img, key, processors, models, rank)  
                
                buf = io.BytesIO()
                np.save(buf, mask_arr.astype(np.int32))  # 保存为 .npy
                buf.seek(0)
                new_item = {
                    "jpg": img,
                    "txt": item['txt'],
                    "small_mask.npy": buf.read(), 
                    #"mask.npy":semantic_mask,  
                    "__key__": key,
                    "__url__": item['__url__']
                    }
                sink.write(new_item)
                processed_imgs.append(key)

            if idx % 500 == 0:
                resume_info = {
                "processed_tars": processed_tars,
                "processed_imgs": processed_imgs,
                "global_classnames": sorted(processors['global_classnames']),
                "label2index": processors['label2index'],
                }
                
                with open(resume_path, "w") as f:
                    json.dump(resume_info, f, indent=2)
                         
        
        processed_tars.append(tar_name)
    if rank == 0:
        with open(os.path.join(output_dir, f"label2index.json"), "w") as f:
            json.dump(processors['label2index'], f, indent=2)


        

if __name__ == '__main__':
    args = parse_args()
    if args.world_size > 1:
        mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)
    else:
        main(0, args)