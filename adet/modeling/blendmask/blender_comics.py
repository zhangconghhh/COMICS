import torch, pdb
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as FF
from detectron2.layers import cat
from detectron2.modeling.poolers import ROIPooler
from .cm_utils import seg2edge, get_query_keys_df
import numpy as np


def build_blender(cfg):
    return Blender(cfg)


class Blender(nn.Module):
    def __init__(self, cfg):
        super(Blender, self).__init__()
        # fmt: off
        self.pooler_resolution = cfg.MODEL.BLENDMASK.BOTTOM_RESOLUTION # 56
        sampling_ratio         = cfg.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO
        pooler_type            = cfg.MODEL.BLENDMASK.POOLER_TYPE
        pooler_scales          = cfg.MODEL.BLENDMASK.POOLER_SCALES
        self.attn_size         = cfg.MODEL.BLENDMASK.ATTN_SIZE
        self.top_interp        = cfg.MODEL.BLENDMASK.TOP_INTERP
        num_bases              = cfg.MODEL.BASIS_MODULE.NUM_BASES
        # fmt: on

        self.attn_len = num_bases * self.attn_size * self.attn_size
        self.pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=2)

        self.T = 0.7  # positive 
        self.TT = 0.7 # negative
        self.conv_layer = nn.Conv2d(128, 128, kernel_size=1)
     

    def calhard(self, masks):
        edges = seg2edge(masks)
        sample_results = get_query_keys_df(edges, masks, thred_u=0.0, scale_u=1.0, percent=1)
        return sample_results


    def calcontrastLoss_intra_cu(self, feats, masks, labels, logits=None):
        feats = feats.view(feats.shape[0], feats.shape[1], -1)
        feats = nn.functional.normalize(feats, dim=1)
        real_feat = feats[labels == 0.0]
        fake_feat = feats[labels == 1.0]
        real_mask = masks[labels == 0.0]
        fake_mask = masks[labels == 1.0]
        if len(real_mask) != 0 and len(fake_mask) != 0:
            real_sample_results = self.calhard(real_mask)
            real_pos_threshold = real_sample_results['easy_positive_sets_N'].view(real_sample_results['easy_positive_sets_N'].shape[0],-1)
            real_neg_threshold = real_sample_results['easy_negative_sets_N'].view(real_sample_results['easy_negative_sets_N'].shape[0],-1)
            real_pos = real_pos_threshold.unsqueeze(1).cuda() * real_feat
            real_neg = real_neg_threshold.unsqueeze(1).cuda() * real_feat
              
            # cal deepfake sim_negtive
            fake_sample_results = self.calhard(fake_mask)
            fake_pos_threshold = fake_sample_results['easy_positive_sets_N'].view(fake_sample_results['easy_positive_sets_N'].shape[0],-1)
            fake_neg_threshold = fake_sample_results['easy_negative_sets_N'].view(fake_sample_results['easy_negative_sets_N'].shape[0],-1)
            fake_pos = fake_pos_threshold.unsqueeze(1).cuda() * fake_feat  
            fake_neg = fake_neg_threshold.unsqueeze(1).cuda() * fake_feat
          
            sim_positive = torch.bmm(real_pos.permute(0, 2, 1), real_neg).view(real_neg.shape[0], -1)  # for real image
            sim_positive /= self.T
            l_real_sim = sim_positive
            l_real_sim = torch.sum(torch.exp(l_real_sim), dim=1)

            sim_negative = torch.bmm(fake_pos.permute(0, 2, 1), fake_neg).view(fake_neg.shape[0], -1)  # for fake part of fake image     
            sim_negative /= self.TT
            l_negative_sim = sim_negative
            l_negative_sim = torch.sum(torch.exp(l_negative_sim), dim=1)    
            loss_fake_intra = -torch.log((torch.sum(l_real_sim)) /
                                            (torch.sum(l_negative_sim) + torch.sum(l_real_sim)))
            return loss_fake_intra
        else: 
            return torch.tensor(0).cuda()


    def calcontrastLoss_intra_cu1(self, feats, masks, labels, gt_instances, logits=None):
        P_sum = [] 
        for x in gt_instances:
            P_sum.append(len(x.gt_boxes)) 
        feats = feats.view(feats.shape[0], feats.shape[1], -1) # feats 31x128x784
        feats = nn.functional.normalize(feats, dim=1)
        num = 0
        loss_fake_intra = torch.cuda.FloatTensor([0])
        for p_num in P_sum:
            feat = feats[num: num+p_num]
            mask = masks[num: num+p_num]
            label = labels[num: num+p_num]        
            index = (feat.sum(dim=1).sum(dim=1)!=0)
            feat = feat[index]
            mask = mask[index]
            label = label[index]
            num += p_num
            real_feat = feat[label == 0.0]
            fake_feat = feat[label == 1.0]
            real_mask = mask[label == 0.0]
            fake_mask = mask[label == 1.0]    
    
            if len(real_mask) != 0 and len(fake_mask) != 0:
                real_sample_results = self.calhard(real_mask)
                real_pos_threshold = real_sample_results['easy_positive_sets_N'].view(real_sample_results['easy_positive_sets_N'].shape[0],-1)
                real_neg_threshold = real_sample_results['easy_negative_sets_N'].view(real_sample_results['easy_negative_sets_N'].shape[0],-1)
                real_pos = real_pos_threshold.unsqueeze(1).cuda() * real_feat
                real_neg = real_neg_threshold.unsqueeze(1).cuda() * real_feat
                # cal deepfake sim_negtive
                fake_sample_results = self.calhard(fake_mask)
                fake_pos_threshold = fake_sample_results['easy_positive_sets_N'].view(fake_sample_results['easy_positive_sets_N'].shape[0],-1)
                fake_neg_threshold = fake_sample_results['easy_negative_sets_N'].view(fake_sample_results['easy_negative_sets_N'].shape[0],-1)
                fake_pos = fake_pos_threshold.unsqueeze(1).cuda() * fake_feat  #1x128x784
                fake_neg = fake_neg_threshold.unsqueeze(1).cuda() * fake_feat
                sim_positive = torch.einsum('ibh,jch->ijbc', real_neg.permute(0,2,1), fake_neg.permute(0,2,1))
                sim_positive  /= self.T
                l_real_sim = torch.sum(torch.exp(sim_positive), dim=1)             
                sim_negative = torch.einsum('ibh,jch->ijbc', real_pos.permute(0,2,1), fake_pos.permute(0,2,1))  
                sim_negative /= self.TT
                l_negative_sim = torch.sum(torch.exp(sim_negative), dim=1) 
                loss_fake_intra += -torch.log((torch.sum(l_real_sim)) /(torch.sum(l_negative_sim) + torch.sum(l_real_sim))).item()
            else: 
                loss_fake_intra += torch.tensor(0).cuda().item()
                
        return loss_fake_intra/len(P_sum)


    def __call__(self, bases,  proposals, gt_instances, basis_fea, con_w):
        if gt_instances is not None:
            dense_info = proposals["instances"]
            attns = dense_info.top_feats
            pos_inds = dense_info.pos_inds
            if pos_inds.numel() == 0:
                return None, {"loss_mask": sum([x.sum() * 0 for x in attns]) + bases[0].sum() * 0}

            gt_inds = dense_info.gt_inds
            rois = self.pooler(bases, [x.gt_boxes for x in gt_instances])
            rois = rois[gt_inds]
            pred_mask_logits = self.merge_bases(rois, attns)
            basis_fea = self.pooler([self.conv_layer(basis_fea)], [x.gt_boxes for x in gt_instances])
            basis_fea = FF.resize(basis_fea,int(self.pooler_resolution/2))
         
            labels,gt_masks = [], []
            for instances_per_image in gt_instances:
                if len(instances_per_image.gt_boxes.tensor) == 0:
                    continue
                gt_mask_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.gt_boxes.tensor, self.pooler_resolution
                ).to(device=pred_mask_logits.device)
                gt_masks.append(gt_mask_per_image)
                labels.append(instances_per_image.gt_classes)
            gt_mask = cat(gt_masks, dim=0)
            gt_masks = gt_mask[gt_inds]
            N = gt_masks.size(0)
            gt_masks = gt_masks.view(N, -1)       
            labels = cat(labels, dim=0) 
      
            gt_ctr = dense_info.gt_ctrs
            loss_denorm = proposals["loss_denorm"]
            mask_losses = F.binary_cross_entropy_with_logits(
                pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="none")
            mask_loss = ((mask_losses.mean(dim=-1) * gt_ctr).sum()
                         / loss_denorm)        

            if con_w ==0: 
                return None, {"loss_mask": mask_loss}
            else:
                loss_contrastive_intra = self.calcontrastLoss_intra_cu(basis_fea, FF.resize(gt_mask.to(int),int(self.pooler_resolution/2)), labels) 
                loss_contrastive_inter =  self.calcontrastLoss_intra_cu1(basis_fea, FF.resize(gt_mask.to(int),int(self.pooler_resolution/2)), labels,gt_instances)               
                return None, {"loss_mask": mask_loss, "loss_fcos_con_intra":loss_contrastive_intra*con_w*5, "loss_fcos_con_img":loss_contrastive_inter*con_w*0.5}
               

        else:
            # no proposals
            total_instances = sum([len(x) for x in proposals])
            if total_instances == 0:
                # add empty pred_masks results
                for box in proposals:
                    box.pred_masks = box.pred_classes.view(
                        -1, 1, self.pooler_resolution, self.pooler_resolution)
                return proposals, {}
            rois = self.pooler(bases, [x.pred_boxes for x in proposals])
            attns = cat([x.top_feat for x in proposals], dim=0)
            pred_mask_logits = self.merge_bases(rois, attns).sigmoid()
            pred_mask_logits = pred_mask_logits.view(
                -1, 1, self.pooler_resolution, self.pooler_resolution)
            start_ind = 0
            for box in proposals:
                end_ind = start_ind + len(box)
                box.pred_masks = pred_mask_logits[start_ind:end_ind]
                start_ind = end_ind
            return proposals, {}

    def merge_bases(self, rois, coeffs, location_to_inds=None):
        # merge predictions
        N = coeffs.size(0)
        if location_to_inds is not None:
            rois = rois[location_to_inds]
        N, B, H, W = rois.size()

        coeffs = coeffs.view(N, -1, self.attn_size, self.attn_size)
        coeffs = F.interpolate(coeffs, (H, W),
                               mode=self.top_interp).softmax(dim=1)
        masks_preds = (rois * coeffs).sum(dim=1)
        return masks_preds.view(N, -1)
        # return rois * coeffs


# utils
@torch.no_grad()
def concat_all_gather(tensor):

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
