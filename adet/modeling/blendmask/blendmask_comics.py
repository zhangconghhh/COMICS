# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch, pdb, copy
import numpy as np
from torch import nn
import torch.nn.functional as F
from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head
from .mvss_utils import SRMFeature, SRMFeature_basis
from adet.layers import conv_with_kaiming_uniform
from .blender import build_blender
from .basis_module import build_basis_module
from .srm_utils import SRMPixelAttention, SRMFeature_mvss

__all__ = ["BlendMask"]


# utils
@torch.no_grad()
def concat_all_gather(tensor):

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



@META_ARCH_REGISTRY.register()
class BlendMask(nn.Module):
    """
    Main class for BlendMask architectures (see https://arxiv.org/abd/1901.02446).
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.instance_loss_weight = cfg.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT
        self.backbone = build_backbone(cfg)
        self.backbone.load_state_dict(torch.load('R_50_1x.pth'), strict=False)
        self.backbone_k = build_backbone(cfg)
        self.backbone_k.load_state_dict(torch.load('R_50_1x.pth'), strict=False)
          
        for param_k in self.backbone_k.parameters():
            param_k.requires_grad = False  # ema update

        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.blender = build_blender(cfg)
        self.basis_module = build_basis_module(cfg, self.backbone.output_shape())

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
        if self.combine_on: #flase
            self.panoptic_module = build_sem_seg_head(cfg, self.backbone.output_shape())
            self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
            self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
            self.combine_instances_confidence_threshold = (
                cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH)

        # build top module
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES
        attn_size = cfg.MODEL.BLENDMASK.ATTN_SIZE
        attn_len = num_bases * attn_size * attn_size
        self.top_layer = nn.Conv2d(
            in_channels, attn_len,
            kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.top_layer.weight, std=0.01)
        torch.nn.init.constant_(self.top_layer.bias, 0)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.con_w = 1.0
        self.m = 0.999
        self.scoreLoss = nn.BCELoss()
        self.SRMlayer_basis = SRMFeature_mvss() 
        self.SRMlayer_basis_k = SRMFeature_mvss() 
        for param_k in self.SRMlayer_basis_k.parameters():
            param_k.requires_grad = False  # ema update
        self.srm_sa = SRMPixelAttention(3)
        self.in_features  = cfg.MODEL.BASIS_MODULE.IN_FEATURES
        self.srm_sa_post = nn.ModuleList()
        for _ in range(5):
            self.srm_sa_post.append(nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)))
        self.srm_sa_post_k = copy.deepcopy(self.srm_sa_post)


    def SRMlayer(self, images):
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = np.asarray([[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]) #shape=(3,3,5,5)

        kernel = torch.FloatTensor(filters).cuda()
        weight = nn.Parameter(data=kernel, requires_grad=False)

        SRM= F.conv2d(images.unsqueeze(dim=0), weight, padding=2).squeeze(0)
        return SRM


    def set_con_avliable(self, con_w=False):
        self.con_w = con_w


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.SRMlayer_basis.parameters(), self.SRMlayer_basis_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.srm_sa_post.parameters(), self.srm_sa_post_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        

    def srm_atten(self, features, atten):        
        features['p3'] = F.interpolate(atten[0], size=(features['p3'].shape[-2], features['p3'].shape[-1]), mode='bilinear', align_corners=True)*features['p3'] + features['p3']
        features['p4'] = F.interpolate(atten[0], size=(features['p4'].shape[-2], features['p4'].shape[-1]), mode='bilinear', align_corners=True)*features['p4'] + features['p4']
        features['p5'] = F.interpolate(atten[0], size=(features['p5'].shape[-2], features['p5'].shape[-1]), mode='bilinear', align_corners=True)*features['p5'] + features['p5']
        features['p6'] = F.interpolate(atten[0], size=(features['p6'].shape[-2], features['p6'].shape[-1]), mode='bilinear', align_corners=True)*features['p6'] + features['p6']
        features['p7'] = F.interpolate(atten[0], size=(features['p7'].shape[-2], features['p7'].shape[-1]), mode='bilinear', align_corners=True)*features['p7'] + features['p7']
        return features
  

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            instances: Instances
            sem_seg: semantic segmentation ground truth.
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: each dict is the results for one image. The dict
                contains the following keys:
                "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                    See the return value of
                    :func:`combine_semantic_and_instance_outputs` for its format.
        """
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
          
        atten =  self.srm_sa(images.tensor)
        features_srm = self.srm_atten(features, atten)
        for i, f in enumerate(self.in_features):
            features_srm[f] = self.srm_sa_post[i](features_srm[f])
      

        if self.training:
            images_k = [x["image1"].to(self.device) for x in batched_inputs]
            images_k = [self.normalizer(x) for x in images_k]
            images_k = ImageList.from_tensors(images_k, self.backbone_k.size_divisibility)
        else:
            images_k = images
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            features_k = self.backbone_k(images_k.tensor)
            features_k_srm = self.srm_atten(features_k, atten)
            for i, f in enumerate(self.in_features):
                features_k_srm[f] = self.srm_sa_post_k[i](features_k_srm[f])

        if self.combine_on:
            if "sem_seg" in batched_inputs[0]:
                gt_sem = [x["sem_seg"].to(self.device) for x in batched_inputs]
                gt_sem = ImageList.from_tensors(
                    gt_sem, self.backbone.size_divisibility, self.panoptic_module.ignore_value
                ).tensor
            else:
                gt_sem = None
            sem_seg_results, sem_seg_losses = self.panoptic_module(features, gt_sem)


        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
       

        proposals, proposal_losses = self.proposal_generator(images, features, features_k, features_srm, features_k_srm, gt_instances, self.top_layer, self.con_w)
        
        if "basis_sem" in batched_inputs[0]:
            basis_sem = [x["basis_sem"].to(self.device) for x in batched_inputs]
            basis_sem = ImageList.from_tensors(
                basis_sem, self.backbone.size_divisibility, 0).tensor
        else:
            basis_sem = None

        basis_out, basis_losses, basis_fea = self.basis_module(features_srm, basis_sem, self.con_w)
        detector_results, detector_losses = self.blender(basis_out["bases"], proposals, gt_instances, basis_fea, self.con_w)
     
        if self.training:
            losses = {}
            losses.update(basis_losses)
            losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
            losses.update(proposal_losses)
        
            if self.combine_on:
                losses.update(sem_seg_losses)
            return losses

        processed_results = []
        for i, (detector_result, input_per_image, image_size) in enumerate(zip(
                detector_results, batched_inputs, images.image_sizes)):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            detector_r = detector_postprocess(detector_result, height, width)
            processed_result = {"instances": detector_r}
            if self.combine_on:  #false
                sem_seg_r = sem_seg_postprocess(
                    sem_seg_results[i], image_size, height, width)
                processed_result["sem_seg"] = sem_seg_r
            if "seg_thing_out" in basis_out: #false
                seg_thing_r = sem_seg_postprocess(
                    basis_out["seg_thing_out"], image_size, height, width)
                processed_result["sem_thing_seg"] = seg_thing_r
            if self.basis_module.visualize: #false
                processed_result["bases"] = basis_out["bases"]
            processed_results.append(processed_result)

            if self.combine_on: #false
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold)
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results
