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
def normalize_class_name(name):
    name = name.lower()
    name = re.sub(r'^(a|an|the) ', '', name)  # remove leading articles
    return name.strip()

# 参数配置
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12306'
#export HF_ENDPOINT="https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

input_dir = '/mnt/33t/cy/blip3o_dataset'
output_dir = '/mnt/33t/cy/mask_dataset'
base_dir = '/mnt/33t/cy/mllm_models/semantic_sam'

os.makedirs(output_dir, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes')
    args = parser.parse_args()
    return args

# ====================== 语义推理函数 ======================
def semantic_infer_one_image(img: Image.Image, processors, models, rank) -> np.ndarray:
    """对一张图像执行语义标注, 返回 shape=(256/w, w) 的单通道 numpy array, 值为 0~num_class"""
    from mmcv import imcrop
    import pycocotools.mask as maskUtils
    #在原始进行ssa的代码里，使用的是 img = mmcv.imread(image_path)的读取方式，此时读取的image格式是bgr的，所以下面的代码中是否需要img = img.convert('BGR') ?
    #但是我查看sam等pretrain model的demo, 图像输入都是rgb类型的, 很疑惑
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
        #这里需要加入对rank的区分, 还是之后统一处理呢？如何能够使得不同rank最后得到的label2index能进行高效的统一编号并保存？
        class_id = processors['label2index'].setdefault(class_name, len(processors['label2index']))
        semantic_mask[valid_mask.numpy()] = class_id

    h, w = semantic_mask.shape
    scale = (256 / (h * w)) ** 0.5
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale)) #如何能保证h*w一定等于256呢？
    mask_img = Image.fromarray(semantic_mask).resize((new_w, new_h), resample=Image.NEAREST) #请给出一个验证程序，计算从resized mask_img还原为semantic_mask后这个转换过程的误差
    mask_arr = np.array(mask_img).astype(np.uint8)

    debug = True
    if debug:



        
        #请提供保存image, image_mask, resized mask, 从resized mask恢复的image mask 到debug_folder中的代码，方便我检查模型效果

    
    return mask_arr  

def main(rank, args):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    device = torch.device(f"cuda:{rank}")
    # 模型与处理器初始化
    processors = {
        'clip': CLIPProcessor.from_pretrained(f"{base_dir}/clip-vit-large-patch14"),
        'clipseg': AutoProcessor.from_pretrained(f"{base_dir}/clipseg-rd64-refined"),
        'oneformer_ade20k': OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large"),
        'oneformer_coco': OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large"),
        'blip': BlipProcessor.from_pretrained(f"{base_dir}/blip-image-captioning-large"),
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
    local_files = tar_files[rank::args.world_size]
    
    for tar_path in (local_files if rank != 0 else tqdm.tqdm(local_files)):
        dataset = load_dataset("webdataset", data_files=tar_path, split="train")
        tar_name = os.path.splitext(os.path.basename(tar_path))[0]
        output_tar_path = os.path.join(output_dir, f"{tar_name}.tar")
        with tarfile.open(output_tar_path, "w") as tar_out:
            for item in dataset:  #{'jpg': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7FD1F1EA6BD0>, 'txt': 'The image depicts a person in military attire, including a helmet and tactical gear, ...ion and movement in a natural environment.', '__key__': 'sa_3631', '__url__': '/mnt/33t/cy/blip3o_dataset/sa_000000.tar'}
                key = item["__key__"]
                img: Image.Image = item["jpg"]
                #for debug
                #img = Image.open('/root/MLLM/Bagel/cy_test/output.png')
                mask_arr = semantic_infer_one_image(img, processors, models, rank)
                
                
                #以下的保存我觉得存在问题，我希望保留item的所有项，然后新增一个'mask'存储mask, mask存储为什么格式最方便？此外其他项如何原样存储？
                img_buffer = io.BytesIO()
                item["image"].save(img_buffer, format="PNG")
                mask_buffer = io.BytesIO(mask_arr.astype(np.uint8).tobytes())

                for name, buf in [("image.png", img_buffer), ("mask.pgm", mask_buffer)]:
                    info = tarfile.TarInfo(f"{key}/{name}")
                    info.size = buf.getbuffer().nbytes
                    buf.seek(0)
                    tar_out.addfile(info, buf)
        if rank == 0:
            print(f"[rank {rank}] Saved: {output_tar_path}")




if __name__ == '__main__':
    args = parse_args()
    if args.world_size > 1:
        mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)
    else:
        main(0, args)