import torch
from torch import nn


def OmniSegLoss(results, target, **kwargs):

    lambda_sam=1e-3 
    sam_level_weight=5
    lambda_depth=1.

    depth_flag = kwargs.get('depth_flag', False)
    semantic_flag = kwargs.get('semantic_flag', True)
    semantic_dim = kwargs.get('semantic_dim', 16)
    semantic_only = kwargs.get('semantic_only', True)
    patch_flag = kwargs.get('patch_flag', False)
    min_pixnum = kwargs.get('min_pixnum', 20)

    patch_flag = True
    normalize_sam = True
    sam_norm_loss_flag = True

    if 'global_step' in kwargs:
        global_step = kwargs['global_step']
    

    d = {}
    if not semantic_only:
        d['rgb'] = (results['rgb']-target['rgb'])**2
        
        if depth_flag and "depth" in target.keys():
            valid_depth_idx = target['depth'] > 0
            d['depth_render'] = lambda_depth * (results['depth'][valid_depth_idx.squeeze()]-target['depth'][valid_depth_idx])**2


    if semantic_flag:

        valid_semantic_idx = target['sam'] > 0
        sam_t = target['sam'][valid_semantic_idx].long()
        sam_o = results['semantic'][valid_semantic_idx.squeeze(), :]
        
        if normalize_sam:
            sam_o = sam_o / (torch.norm(sam_o, dim=-1, keepdim=True) + 1e-6).detach()
        

        ## --- Contructive Clustering --- #
        cluster_ids, cnums_all = torch.unique(sam_t, return_counts=True)
        cluster_ids = cluster_ids[cnums_all > min_pixnum]
        cnums = cnums_all[cnums_all > min_pixnum]
        cnum = cluster_ids.shape[0] # cluster number

        u_list = torch.zeros([cnum, sam_o.shape[-1]], dtype=torch.float32, device=sam_o.device)
        phi_list = torch.zeros([cnum, 1], dtype=torch.float32, device=sam_o.device)


        for i in range(cnum):
            cluster = sam_o[sam_t == cluster_ids[i], :]
            u_list[i] = torch.mean(cluster, dim=0, keepdim=True)
            phi_list[i] = torch.norm(cluster - u_list[i], dim=1, keepdim=True).sum() / (cnums[i] * torch.log(cnums[i] + 10))

        if patch_flag:
            accpatch = target['accpatch']


        # tau = 0.1; phi_list[:, 0] = tau    # option 1: constant temperature
        # phi_list = phi_list * (tau / phi_list.mean())     # option 2: (PCL) too small phi causes too large num in torch.exp().
        # phi_list = (phi_list - phi_list.min()) / (phi_list.max() - phi_list.min()) * 5 + 0.1   # scale to range [0.1, 5.1]
        phi_list = torch.clip(phi_list * 10, min=0.5, max=1.0)
        phi_list = phi_list.detach()
        
        ProtoNCE = torch.zeros([1], dtype=torch.float32, device=sam_o.device)

        for i in range(cnum):
            cluster = sam_o[sam_t == cluster_ids[i], :]

            dist = torch.exp(torch.matmul(cluster, u_list.T) / phi_list.T)  # [N_pix, N_cluster]

            if not patch_flag:

                ProtoNCE += -torch.sum(torch.log(
                    dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                    ))

            else:
                acc_list = accpatch[cluster_ids[i], :]
                acc_list_c = acc_list[cluster_ids]  # for the clusters only
                acc_h_cnts = torch.sort(torch.unique(acc_list_c), descending=True).values[:-1]    # remove last one "0"
                levelnum = acc_h_cnts.shape[0]
                
                for l, level in enumerate(acc_h_cnts):
                    level_cids = torch.argwhere(acc_list_c == level).squeeze()   # cluster ids on each level

                    cal_opt = 3
                    
                    if cal_opt == 1:
                        # --- option 1: mean patches dist
                        dist_mean = dist[:, level_cids].reshape(dist.shape[0], -1).mean(dim=1, keepdim=True)
                        tmp_loss = -torch.sum(torch.log(
                                dist_mean / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                            ))
                        max_loss = tmp_loss if l == 0 else max(tmp_loss, max_loss)  # --- unidirectional hierarchical loss --- #

                    elif cal_opt == 2:
                        # --- option 2: all patches dist
                        dist_patches = dist[:, level_cids].reshape(dist.shape[0], -1)
                        tmp_loss = -torch.sum(
                                torch.log(
                                    dist_patches / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                                ).mean(dim=1)
                            )
                        max_loss = tmp_loss if l == 0 else max(tmp_loss, max_loss)  # --- unidirectional hierarchical loss --- #
                    
                    elif cal_opt == 3:
                        # --- option 3: per pixel patches dist,  unidirectional hierarchical loss --- #
                        dist_patches = dist[:, level_cids].reshape(dist.shape[0], -1)

                        log = - torch.log(
                                dist_patches / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)     # (pixnum, L_patchnum)
                            )
                        if l == 0:
                            max_loss = torch.sum( log.mean(dim=1) )
                            # --- set max_log for each pix --- #
                            max_log = torch.max(log, dim=1, keepdim=True).values   # patch with max lose (pixnum, 1)
                        else:
                            # --- thres by the last layer --- #
                            log_thres = torch.where(log > max_log, log, max_log)
                            max_loss = torch.sum( log_thres.mean(dim=1) )
                            # --- update max_log for each pix --- #
                            max_log = torch.max(log_thres, dim=1, keepdim=True).values   # patch with max lose (pixnum, 1)
                    
                    # ProtoNCE += max_loss
                    # ProtoNCE += max_loss / levelnum
                    ProtoNCE += max_loss * sam_level_weight**(-l)     # --- level weighting --- #

        d['semantic_render'] = lambda_sam * ProtoNCE

        # if sam_norm_loss_flag:
        #     sam_norm_loss = ((torch.norm(sam_o, dim=-1, keepdim=True) - 1.0) ** 2).mean()
        #     d['sam_norm_loss'] = 1000 * sam_norm_loss


    return d