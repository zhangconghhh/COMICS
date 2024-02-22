import math, pdb, copy
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from .fcos_outputs_comics import FCOSOutputs
from .comics_utils import SRMFeature_FCOS

__all__ = ["FCOS"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS.YIELD_PROPOSAL
        self.yield_box_feats = cfg.MODEL.FCOS.YIELD_BOX_FEATURES

        self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module

        self.fcos_outputs = FCOSOutputs(cfg)
       
        self.SRMFeature_FCOS = SRMFeature_FCOS()
        self.fine = nn.ModuleList()
        in_channels = 256
        for ith in range(3):
            if ith == 2:
                self.fine.append(nn.Conv2d(in_channels, in_channels, kernel_size=1))
            else:
                # self.fine.append(nn.Conv2d(in_channels, in_channels,kernel_size=1,bias=True))
                self.fine.append(nn.Conv2d(in_channels, in_channels, kernel_size=1))
                self.fine.append(nn.ReLU())
            
        self.fine_srm = copy.deepcopy(self.fine)

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

    def forward(self, images, features, features_k, features_srm, features_k_srm,  gt_instances=None, top_module=None, con_w=0):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        features_k = [features_k[f] for f in self.in_features]

        features_srm = [features_srm[f] for f in self.in_features]
        features_k_srm = [features_k_srm[f] for f in self.in_features]

        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers, features_con, feature_con_k, length = self.fcos_head(
            features, features_k, features_srm, features_k_srm, top_module, self.yield_proposal or self.yield_box_feats
        )
    
        if self.training:
            results, losses = self.fcos_outputs.losses(
                logits_pred, reg_pred, ctrness_pred,
                locations, gt_instances, top_feats, features_con, feature_con_k, con_w, length
            )

            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.fcos_outputs.predict_proposals(
                        logits_pred, reg_pred, ctrness_pred,
                        locations, images.image_sizes, top_feats
                    )
            if self.yield_box_feats:
                results["box_feats"] = {
                    f: b for f, b in zip(self.in_features, bbox_towers)
                }
            return results, losses
        else:
            results = self.fcos_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, images.image_sizes, top_feats
            )
            extras = {}
            if self.yield_box_feats:
                extras["box_feats"] = {
                    f: b for f, b in zip(self.in_features, bbox_towers)
                }
            return results, extras

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                                cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                                 cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                                  False),
                        "share_srm": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                        False)}
        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                if head == "cls" and i ==0 : 
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                    ))
                else:
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                    ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))
                        
        self.cls_logits = nn.Conv2d(in_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.ctrness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None
        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness, self.share_srm_tower
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        self.m = 0.999        
        self.refine = nn.ModuleList()
        for ith in range(5):
            self.refine.append(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        self.refine_k = copy.deepcopy(self.refine)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.refine.parameters(), self.refine_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.share_tower.parameters(), self.share_tower_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
     

    def forward(self, x, x_k, features_srm, features_k_srm, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        features = []
        features_k = []
        length = []
               
        for l, feature in enumerate(features_srm):         
            features.append(self.refine[l](feature))
            length.append([feature.shape[-1],feature.shape[-2],feature.shape[0]])      
        for l, feature in enumerate(features_k_srm):
            features_k.append(self.refine_k[l](feature))
           
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)    
            feature_srm = self.share_srm_tower(features_srm[l])    
            cls_tower = self.cls_tower(feature_srm)          
            bbox_tower = self.bbox_tower(feature_srm)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)
            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
          
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))


        return logits, bbox_reg, ctrness, top_feats, bbox_towers, features, features_k, length
