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

input_dir = '/ssdwork/chengyu/blip3o_dataset'
output_dir = '/ssdwork/chengyu/mask_dataset'
base_dir = '/ssdwork/chengyu/mllm_models/semantic_sam'

debug = True
if debug:
    save_key = ""
    debug_folder = "/ssdwork/chengyu/mask_debug_rgb"
    os.makedirs(debug_folder, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes')
    args = parser.parse_args()
    return args

def semantic_infer_one_image(img: Image.Image, processors, models, rank, preprocess = False):
    """对一张图像执行语义标注, 返回 shape=(256/w, w) 的单通道 numpy array, 值为 0~num_class"""
    from mmcv import imcrop
    import pycocotools.mask as maskUtils
    img_arr = np.array(img)
    anns = {'annotations': processors['sam_generator'].generate(img_arr)}
    class_ids_from_oneformer_coco = oneformer_coco_segmentation(img,
                                                                 processors['oneformer_coco'], models['oneformer_coco'], rank)
    class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(img,
                                                                     processors['oneformer_ade20k'], models['oneformer_ade20k'], rank)
    semantic_mask = np.zeros(img.size, dtype=np.uint8)
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
        
        #class_id = processors['label2index'].setdefault(class_name, len(processors['label2index']))
        #semantic_mask[valid_mask.numpy()] = class_id
        processors['local_classnames'].add(class_name)
        if preprocess:
            return 
        class_id = processors['label2index'][class_name]
        semantic_mask[valid_mask.numpy()] = class_id

    h, w = semantic_mask.shape
    scale = (256 / (h * w)) ** 0.5
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale)) #如何能保证h*w一定等于256呢？
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
        #请提供保存image, image_mask, resized mask, 从resized mask恢复的image mask 到debug_folder中的代码，方便我检查模型效果
        
        def save_debug_images(debug_dir, key, image, mask_arr, semantic_mask):
            os.makedirs(debug_dir, exist_ok=True)
            orig_mask_img = Image.fromarray(semantic_mask.astype(np.uint16), mode='I;16')
            resized_mask_img = Image.fromarray(mask_arr.astype(np.uint16), mode='I;16')
            recovered_mask_img = resized_mask_img.resize(semantic_mask.shape[::-1], resample=Image.NEAREST)

            image.save(os.path.join(debug_dir, f"{key}_image.jpg"))
            orig_mask_img.save(os.path.join(debug_dir, f"{key}_mask_orig.png"))
            resized_mask_img.save(os.path.join(debug_dir, f"{key}_mask_resized.png"))
            recovered_mask_img.save(os.path.join(debug_dir, f"{key}_mask_recovered.png"))
        save_debug_images(debug_folder, save_key, img, mask_arr, semantic_mask)
    
    return mask_arr  

def main(rank, args):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    device = torch.device(f"cuda:{rank}")
    processors = {
        'clip': CLIPProcessor.from_pretrained(f"{base_dir}/clip-vit-large-patch14"),
        'clipseg': AutoProcessor.from_pretrained(f"{base_dir}/clipseg-rd64-refined"),
        'oneformer_ade20k': OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large"),
        'oneformer_coco': OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large"),
        'blip': BlipProcessor.from_pretrained(f"{base_dir}/blip-image-captioning-large"),
        'local_classnames': set(),
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
    tar_files = tar_files[:2]
    local_files = tar_files[rank::args.world_size]
    
    
    # Dummy run to collect local class names (skip writing)
    for tar_path in local_files:
        dataset = load_dataset("webdataset", data_files=tar_path, split="train")
        for idx, item in enumerate(tqdm.tqdm(dataset) if rank == 0 else dataset):
            if idx > 200:
                break
            _ = semantic_infer_one_image(item["jpg"], processors, models, rank, preprocess = True)
    
    gathered_classnames = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered_classnames, list(processors['local_classnames']))
    all_classnames = sorted(set().union(*[set(lst) for lst in gathered_classnames]))
    processors['label2index'] = {name: idx for idx, name in enumerate(all_classnames)}
    if rank == 0:
        import json
        with open(os.path.join(output_dir, "label2index.json"), 'w') as f:
            json.dump(processors["label2index"], f, indent=2)
    for tar_path in (local_files if rank != 0 else tqdm.tqdm(local_files)):
        dataset = load_dataset("webdataset", data_files=tar_path, split="train")
        tar_name = os.path.splitext(os.path.basename(tar_path))[0]
        output_tar_path = os.path.join(output_dir, f"{tar_name}.tar")
        if os.path.exists(output_tar_path):
            continue
        with wds.ShardWriter(output_tar_path) as sink:
            for idx, item in enumerate(dataset):
                if idx > 200:
                    break
                key = item["__key__"]
                save_key = key
                img: Image.Image = item["jpg"]
                mask_arr = semantic_infer_one_image(img, processors, models, rank)  
                
                buf = io.BytesIO()
                np.save(buf, mask_arr.astype(np.int32))  # 保存为 .npy
                buf.seek(0)
                new_item = {
                    "jpg": img,
                    "txt": item['txt'],
                    "mask.npy": buf.read(),  
                    "__key__": key,
                    "__url__": item['url']
                }
            sink.write(new_item)
        



    # 还需要修改和提升的功能：
    # 1.label2index 应考虑在多 GPU 情况下共享
    # 2. 保存 label2index 为 JSON 方便后续 decode 使用
    # 3. 支持 resume（跳过已生成 tar）以便中断恢复


     
    
if __name__ == '__main__':
    args = parse_args()
    if args.world_size > 1:
        mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)
    else:
        main(0, args)