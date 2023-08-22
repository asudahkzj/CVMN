"""
CVMN model and criterion classes.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from cmath import isnan
from numpy import dtype
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .segmentation import (CVMNsegm, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer

import math



class CVMN(nn.Module):
    def __init__(self, backbone, transformer, num_frames, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.num_frames = num_frames
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.proj_t = nn.Conv1d(768, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.hallucinator = MLP(hidden_dim, hidden_dim, 1024, 2)

    def forward(self, samples: NestedTensor, expressions):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # moved the frame to batch dimension for computation efficiency
        features, pos = self.backbone(samples)
        pos = pos[-1]
        src, mask = features[-1].decompose()
        src_proj = self.input_proj(src)
        n,c,h,w = src_proj.shape
        assert mask is not None
        src_proj = src_proj.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(0,2,1,3,4).flatten(-2)
        mask = mask.reshape(n//self.num_frames, self.num_frames, h*w)
        pos = pos.permute(0,2,1,3,4).flatten(-2)

        exp = self.embedding(expressions)

        hs = self.transformer(src_proj, mask, exp, self.query_embed.weight, pos)[0]

        # outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out = {'pred_boxes': outputs_coord[-1], 'conv_audio': exp}
        if self.aux_loss:
            out['aux_outputs'] = outputs_coord[:-1]
            # out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for CVMN.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, weight_dict, eos_coef, losses, num_frames):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        # self.num_classes = num_classes
        # self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_frames = num_frames


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        # src_boxes = outputs['pred_boxes'][idx]
        src_boxes = outputs['pred_boxes']
        # if targets[0]['boxes'].shape[0] != self.num_frames:
        #     src_boxes = src_boxes[:, (self.num_frames-1)//2, :].unsqueeze(1)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        src_boxes = src_boxes[idx]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], split=False).decompose()
        # if target_masks.shape[1] != self.num_frames:
        #     src_masks = src_masks[:, (self.num_frames-1)//2, :, :].unsqueeze(1)
        target_masks = target_masks.to(src_masks)
        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        try: 
            src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
            src_masks = src_masks[:, 0].flatten(1)
            target_masks = target_masks[tgt_idx].flatten(1)
        except:
            src_masks = src_masks.flatten(1)
            target_masks = src_masks.clone()
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_kl(self, outputs, targets, indices, num_boxes):
        
        assert "memory" in outputs
        assert "fusion" in outputs

        src_mem = outputs['memory']
        tgt_fus = outputs['fusion']
        # loss_kl = F.kl_div(src_mem, tgt_fus, reduction='none')
        # loss_kl = F.kl_div(src_mem, tgt_fus, reduction='batchmean')
        logp_src_mem = F.log_softmax(src_mem, dim=-1)
        p_tgt_fus = F.softmax(tgt_fus, dim=-1)
        loss_kl = F.kl_div(logp_src_mem, p_tgt_fus)

        # src_mem = src_mem.permute(1, 0, 2)[0]
        # tgt_fus = tgt_fus.permute(1, 0, 2)[0]
        # logp_src_mem = F.log_softmax(src_mem, dim=-1)
        # p_tgt_fus = F.softmax(tgt_fus, dim=-1)
        # loss_kl = F.kl_div(logp_src_mem, p_tgt_fus)

        losses = {
            "loss_kl": loss_kl,
        }
        return losses

    def loss_p(self, outputs, targets, indices, num_boxes):
        
        assert "memory_h" in outputs
        assert "video_concept" in outputs
        assert "memory_h_t" in outputs
        assert "video_concept_t" in outputs

        src_mem_s = outputs['memory_h'][0]
        tgt_vc_s = outputs['video_concept']

        logp_src_mem_s = F.log_softmax(src_mem_s, dim=-1)
        p_tgt_vc_s = F.softmax(tgt_vc_s, dim=-1)
        loss_ps = F.kl_div(logp_src_mem_s, p_tgt_vc_s)

        src_mem_t = outputs['memory_h_t'][0]
        tgt_vc_t = outputs['video_concept_t']

        logp_src_mem_t = F.log_softmax(src_mem_t, dim=-1)
        p_tgt_vc_t = F.softmax(tgt_vc_t, dim=-1)
        loss_pt = F.kl_div(logp_src_mem_t, p_tgt_vc_t)

        losses = {
            "loss_ps": loss_ps,
            "loss_pt": loss_pt,
        }
        return losses

    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)


    def loss_mmd(self, outputs, targets, indices, num_boxes):
        
        assert "memory_h" in outputs
        assert "memory_h_t" in outputs

        mmd_batch = outputs['mmd_batch']
        accumulation_steps = outputs['accumulation_steps']
        if len(mmd_batch) < accumulation_steps:
            loss_mmd = torch.tensor(0, dtype=torch.float32, device=mmd_batch[0][0].device)
        else:
            mem_s, mem_t = zip(*mmd_batch)
            im_s = torch.cat(mem_s[:accumulation_steps-1], 0).data
            mem_s = torch.cat((im_s, mem_s[accumulation_steps-1]), 0).flatten(-2)
            im_t = torch.cat(mem_t[:accumulation_steps-1], 0).data
            mem_t = torch.cat((im_t, mem_t[accumulation_steps-1]), 0).flatten(-2)
            # mem_s = torch.cat(mem_s, 0).flatten(-2)
            # mem_t = torch.cat(mem_t, 0).flatten(-2)
            delta = mem_s - mem_t
            # loss_mmd1 = mmd_linear(mem_s, mem_t)
            # loss_mmd = mmd_rbf(mem_s, mem_t)*20 + mmd_linear(mem_s, mem_t)*0.5
            loss_mmd = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
            loss_mmd = loss_mmd * accumulation_steps

        # mem_s = outputs['memory_h']
        # mem_t = outputs['memory_h_t']
       
        # mem_s = mem_s.squeeze(0)
        # mem_t = mem_t.squeeze(0)

        # delta = mem_s - mem_t
        # # a = torch.mm(delta, torch.transpose(delta, 0, 1))
        # # loss_mmd = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        # # a = mmd_rbf(mem_s, mem_t)
        # # b = mmd_linear(mem_s, mem_t) 
        # loss_mmd = mmd_rbf(mem_s, mem_t)*0.5 + mmd_linear(mem_s, mem_t)*0.5
        if torch.isnan(loss_mmd):
            loss_mmd = torch.tensor(0, dtype=torch.float32, device=loss_mmd.device)

        losses = {
            "loss_mmd": loss_mmd,
        }
        return losses

    def loss_re(self, outputs, targets, indices, num_boxes):
        assert "cand_text" in outputs
        assert "rec_feature" in outputs

        assert "pseudo_id" in outputs
        pseudo_id = outputs['pseudo_id']

        temperature = 0.5

        cand_text = outputs['cand_text'].float()
        rec_feature_s = outputs['rec_feature_s'].float()
        rec_feature_s = torch.mean(rec_feature_s, dim=0, keepdim=True)
        rec_feature_t = outputs['rec_feature_t'].float()
        rec_feature_t = torch.mean(rec_feature_t, dim=0, keepdim=True)

        # image_features = rec_feature / rec_feature.norm(dim=1, keepdim=True)
        # text_features = cand_text / cand_text.norm(dim=1, keepdim=True)
        # import numpy as np
        # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        # logits_per_image = image_features @ text_features.t()
        # probs = logits_per_image.softmax(dim=-1)

        # z_ori = F.normalize(cand_text, dim=1)
        # z_re_s = F.normalize(rec_feature, dim=1)
        sim_s = F.cosine_similarity(cand_text, rec_feature_s)
        nominator_s = torch.exp(sim_s[0] / temperature)
        denominator_s = torch.exp(sim_s / temperature)
        cont_loss_s = -torch.log(nominator_s / torch.sum(denominator_s))
        if torch.isnan(cont_loss_s):
            import numpy as np
            print('cand_text:', cand_text)
            np.save('cand_text.npy', cand_text.cpu().numpy())
            print('rec_feature_s:', rec_feature_s)
            np.save('rec_feature_s.npy', rec_feature_s.cpu().numpy())
            print('sim_s:', sim_s)
            print('nominator_s', nominator_s)
            print('denominator_s', denominator_s)
            cont_loss_s = torch.tensor(0, dtype=torch.float32, device=cont_loss_s.device)
        
        sim_t = F.cosine_similarity(cand_text, rec_feature_t)
        # nominator_t = torch.exp(sim_t[0] / temperature)
        nominator_t = torch.exp(sim_t[pseudo_id] / temperature)
        denominator_t = torch.exp(sim_t / temperature)
        cont_loss_t = -torch.log(nominator_t / torch.sum(denominator_t))
        if torch.isnan(cont_loss_t):
            cont_loss_t = torch.tensor(0, dtype=torch.float32, device=cont_loss_t.device)

        losses = {
            "loss_res": cont_loss_s,
            "loss_ret": cont_loss_t,
        }
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            # 'labels': self.loss_labels,
            # 'cardinality': self.loss_cardinality,
            're':self.loss_re,
            'mmd':self.loss_mmd,
            'p':self.loss_p,
            'kl':self.loss_kl,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)
        indices = []
        bs = outputs['pred_boxes'].shape[0]
        for i in range(bs):
            valid = targets[i]['valid']
            index_i,index_j = [],[]
            for j in range(len(valid)):
                if valid[j] == 1:
                    index_i.append(j)
                    index_j.append(j)
            index_i = torch.tensor(index_i).long().to(valid.device)
            index_j = torch.tensor(index_j).long().to(valid.device)
            indices.append((index_i, index_j))

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 're':
                        continue
                    if loss == 'mmd':
                        continue
                    if loss == 'p':
                        continue
                    if loss == 'kl':
                        continue
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        # return self.lut(x) * math.sqrt(self.d_model)
        return self.lut(x)


def build(args):
    # if args.dataset_file == "ytvos":
    #     num_classes = 41
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    # f = open('wenet/train.yaml', 'r', encoding='utf-8')
    # cfg = f.read()
    # cfg = yaml.load(cfg)
    # wenet = build_wenet(cfg)
    # wenet.load_state_dict(torch.load('wenet/final.pt'))

    model = CVMN(
        backbone,
        transformer,
        # num_classes=num_classes,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = CVMNsegm(model)
    weight_dict = {'loss_res':0.007, 'loss_ret':0.003, 'loss_mmd':0.1, 'loss_ps':500, 'loss_pt':500, 'loss_kl': 500, 'loss_bbox': args.bbox_loss_coef} # 5
    weight_dict['loss_giou'] = args.giou_loss_coef # 2
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef # 1
        weight_dict["loss_dice"] = args.dice_loss_coef # 1
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    # weight_dict1 = {}
    # for k, v in weight_dict.items():
    #     weight_dict1[k+'_s'] = v
    #     weight_dict1[k+'_t'] = v

    losses = ['re', 'mmd', 'p', 'kl', 'boxes']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(weight_dict=weight_dict, eos_coef=args.eos_coef,
                             losses=losses, num_frames=args.num_frames)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
    return model, criterion, postprocessors
