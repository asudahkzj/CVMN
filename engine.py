"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import copy
from importlib.resources import is_resource
import math
import os
import sys
from typing import Iterable
from cv2 import accumulate

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

import torchvision.models as models
import clip

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    source_loader: Iterable, target_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # wenet.evaluate()
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    count = 0

    target_iter = iter(target_loader)
    num_iter = len(target_loader)

    selector, preprocess = clip.load("RN50", device=device)

    accumulation_steps = 4
    mmd_batch = []
    acc_loss = 0

    for samples_s, expressions_s, targets_s, cd_s in metric_logger.log_every(source_loader, print_freq, header):
        count += 1

        samples_t, _, _, cd_t = target_iter.next()
        if count % num_iter == 0:
            target_iter = iter(target_loader)
            
        samples_s = samples_s.to(device)
        expressions_s = expressions_s.to(device)
        targets_s = [{k: v.to(device) for k, v in t.items()} for t in targets_s]
        img_clip_s = cd_s[0][0].to(device)
        text_clip_s = cd_s[0][1].to(device)

        samples_t = samples_t.to(device)
        img_clip_t = cd_t[0][0].to(device)

        with torch.no_grad():
            video_concept_s = selector.encode_image(img_clip_s).float()   # 36*3*224*224 -> 36*512
            video_concept_t = selector.encode_image(img_clip_t).float()
            logits_per_image, logits_per_text = selector(img_clip_t, text_clip_s)
            score_per_text = torch.mean(logits_per_text, dim=1)
            probs = score_per_text.softmax(dim=-1)
            tid = torch.argmax(probs)

        exp_tensor, exp_mask = expressions_s.decompose()
        expressions_s = exp_tensor[0][:exp_mask.shape[1]-exp_mask[0].sum()].unsqueeze(0)
        expressions_t = exp_tensor[tid][:exp_mask.shape[1]-exp_mask[tid].sum()].unsqueeze(0)

        outputs_s = model(samples_s, expressions_s)
        outputs_t = model(samples_t, expressions_t)
        outputs_s['video_concept'] = video_concept_s
        outputs_s['video_concept_t'] = video_concept_t
        # batch_accumulation
        mmd_batch.append((outputs_s['memory_h'], outputs_t['memory_h']))
        outputs_s['mmd_batch'] = mmd_batch
        outputs_s['accumulation_steps'] = accumulation_steps

        outputs_s['memory_h_t'] = outputs_t['memory_h']
        outputs_s['memory_t'] = outputs_t['memory']

        # rec
        rec_feature_s = outputs_s['rec_feature']
        rec_feature_t = outputs_t['rec_feature']
        with torch.no_grad():
            cand_text = selector.encode_text(text_clip_s)
            rec_feature_s = selector.encode_image(rec_feature_s)
            rec_feature_t = selector.encode_image(rec_feature_t)
            outputs_s['rec_feature_s'] = rec_feature_s
            outputs_s['rec_feature_t'] = rec_feature_t
            outputs_s['cand_text'] = cand_text
            outputs_s['pseudo_id'] = tid

        # loss_dict = {}
        loss_dict = criterion(outputs_s, targets_s)

        # for k, v in loss_dict_s.items():
        #     loss_dict[k+'_s'] = v
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # losses.requires_grad_(True)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # batch accumulation
        if count == 1:
            optimizer.zero_grad()
        losses = losses / accumulation_steps
        # acc_loss += losses
        losses.backward()
        if count % accumulation_steps == 0:
            # losses.backward()
            # print(count, 'ba')
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
            mmd_batch = []
            # acc_loss = 0
        # else:
        #     losses.backward(retain_graph=True)

        # optimizer.zero_grad()
        # losses.backward()
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(class_error=0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    print('11111111111111')
    metric_logger.synchronize_between_processes()
    print('22222222222222')
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
