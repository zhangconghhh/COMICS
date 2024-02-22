import torch, pdb
import numpy as np
from vlkit.dense import seg2edge as vlseg2edge
import numpy as np
import torch
import random
from scipy.ndimage.morphology import distance_transform_edt
import torch.nn.functional as F

def seg2edge(seg):
    bs = seg.size(0)
    segnp = seg.cpu().numpy()
    edge = np.zeros_like(segnp)
    for i in range(bs):
        edge[i] = vlseg2edge(segnp[i])
    return torch.tensor(edge).to(seg)



def get_pixel_sets_distrans(src_sets, radius=2):
    """
        src_sets: shape->[N, 28, 28]
    """
    if isinstance(src_sets, torch.Tensor):
        src_sets = src_sets.numpy()
    if isinstance(src_sets, np.ndarray):
        keeps =[]
        for src_set in src_sets:
            keep = distance_transform_edt(np.logical_not(src_set))
            keep = keep < radius
            keeps.append(keep.astype(np.float))
    else:
        raise ValueError(f'only np.ndarray is supported!')
    return torch.tensor(keeps).to(dtype=torch.long)


def get_pixel_sets_N(src_sets, select_num):
    return_ = []
    if isinstance(src_sets, torch.Tensor):
        bs, h,w = src_sets.shape
        keeps_all = torch.where(src_sets>0.5, 1, 0).reshape(bs,-1)
        for idx,keeps in enumerate(keeps_all):
            keeps_init = np.zeros_like(keeps)
            src_set_index = np.arange(len(keeps))
            src_set_index_keeps = src_set_index[keeps.numpy().astype(np.bool)]
            resultList=random.sample(range(0,len(src_set_index_keeps)),int(select_num[idx]))
            src_set_index_keeps_select = src_set_index_keeps[resultList]
            keeps_init[src_set_index_keeps_select]=1
            return_.append(keeps_init.reshape(h,w))
    else:
        raise ValueError(f'only tensor is supported!')
    return torch.tensor(return_) * src_sets



def get_query_keys_df(
                # cams,
                edges,
                masks=None,
                # is_novel=None,
                thred_u=0.1,
                scale_u=1.0,
                percent=0.3):
    #######################################################
    #---------- some pre-processing -----------------------
    #######################################################
    # cams = cams.squeeze(1).cpu() #to cpu, Nx28x28
    edges = edges.cpu() #to cpu, Nx28x28
    masks = masks.cpu() #to cpu, Nx28x28

    #---------- get query mask for each proposal ----------
    query_pos_sets = masks.to(dtype=torch.bool)  #here, pos=foreground area  neg=background area
    query_neg_sets = torch.logical_not(query_pos_sets)

    #For base(seen), get keys according to gt_mask and edges
    edge_sets_dilate = get_pixel_sets_distrans(edges, radius=2)  #expand edges with radius=2
    # pdb.set_trace()
    hard_pos_neg_sets = edge_sets_dilate - edges   #hard keys for both pos and neg
    hard_negative_sets = torch.where((hard_pos_neg_sets - masks)>0.5, 1.0, 0.0)
    hard_positive_sets = torch.where((hard_pos_neg_sets - hard_negative_sets)>0.5, 1.0, 0.0)
    easy_positive_sets = torch.where((masks - hard_pos_neg_sets)>0.5, 1.0, 0.0)
    easy_negative_sets = torch.logical_not(torch.where((masks + edge_sets_dilate) > 0.5 ,1.0, 0.0)).to(dtype=easy_positive_sets.dtype)
 
    return_result = dict()
    return_result['hard_positive_sets_N'] = hard_positive_sets.to(dtype=torch.bool)
    return_result['hard_negative_sets_N'] = hard_negative_sets.to(dtype=torch.bool)

    return_result['easy_positive_sets_N'] = easy_positive_sets.to(dtype=torch.bool) #+ hard_positive_sets.to(dtype=torch.bool)
    return_result['easy_negative_sets_N'] = easy_negative_sets.to(dtype=torch.bool) #+ hard_negative_sets.to(dtype=torch.bool)


    # all
    # return_result['hard_positive_sets_N'] = hard_positive_sets.to(dtype=torch.bool)+easy_positive_sets.to(dtype=torch.bool)
    # return_result['hard_negative_sets_N'] = hard_negative_sets.to(dtype=torch.bool)+easy_negative_sets.to(dtype=torch.bool)

    
    

    return return_result




def get_query_keys_auto(scores, masks):
    scores = torch.sigmoid(scores)
    scores = scores.cpu() #to cpu, Nx28x28
    masks = masks.cpu() #to cpu, Nx28x28
    easy_positive_sets = torch.where(scores>0.65, 1.0, 0.0).to(dtype=torch.bool) &  torch.where(masks==1, 1.0, 0.0).view(masks.shape[0], -1).to(dtype=torch.bool) 
    easy_negative_sets = torch.where(0.35>scores, 1.0, 0.0).to(dtype=torch.bool) &  torch.where(masks==0, 1.0, 0.0).view(masks.shape[0], -1).to(dtype=torch.bool) 

    return_result = dict()   
    return_result['easy_positive_sets_N'] = easy_positive_sets
    return_result['easy_negative_sets_N'] = easy_negative_sets

    return return_result
