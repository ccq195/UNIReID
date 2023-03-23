from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100
    return all_cmc, mAP, indices


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, set=0, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        if set == 2:
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
        else:
            orig_cmc = matches[q_idx]

        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc* 100, mAP* 100, mINP* 100

class Evaluator():
    def __init__(self, args, img_loader, txt_loader, sketch_loader, test_setting=0):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.sketch_loader = sketch_loader # query2
        self.args = args

        self.test_setting = test_setting
        self.logger = logging.getLogger("CLIP2ReID.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, qids_sketch, gids, qfeats_text, qfeats_sketch, qfeats_text_sketch, gfeats, qimage_ids, qimage_ids_sketch, gimage_ids = [], [], [], [], [], [], [], [], [], []

        # text+sketch
        for simg, simage_id, pid, caption in self.txt_loader:
            caption = caption.to(device)
            simg = simg.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                sketch_feat = model.encode_image(simg)
                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    text_sketch_fu = model.fusion_layer(text_feat, sketch_feat, caption, way=self.args.fusion_way)
                    # text_sketch_fu = model.fusion_layer(text_feat, sketch_feat, caption, pa=self.args.pa, way=self.args.fusion_way)
                else:
                    text_sketch_fu = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float()
                text_feat = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float()
                
            qids.append(pid.view(-1)) # flatten 
            qfeats_text.append(text_feat)
            qfeats_text_sketch.append(text_sketch_fu)
            qimage_ids.append(simage_id)
            

        qids = torch.cat(qids, 0)
        qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_text_sketch = torch.cat(qfeats_text_sketch, 0)
        qimage_ids = torch.cat(qimage_ids, 0)

        # image
        for pid, img, image_id in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)[:, 0, :].float()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
            gimage_ids.append(image_id)

        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        gimage_ids = torch.cat(gimage_ids, 0)

        # sketch
        for pid, simg, simage_id in self.sketch_loader:
            simg = simg.to(device)
            with torch.no_grad():
                simg_feat = model.encode_image(simg)[:, 0, :].float()
            qids_sketch.append(pid.view(-1)) # flatten
            qfeats_sketch.append(simg_feat) 
            qimage_ids_sketch.append(simage_id)

        qids_sketch = torch.cat(qids_sketch, 0)
        qfeats_sketch = torch.cat(qfeats_sketch, 0)
        qimage_ids_sketch = torch.cat(qimage_ids_sketch, 0)

        return qfeats_text, qfeats_sketch, qfeats_text_sketch, gfeats, qids, qids_sketch, gids, qimage_ids, qimage_ids_sketch, gimage_ids
    
    def eval(self, model, i2t_metric=False):

        qfeats_text, qfeats_sketch, qfeats_text_sketch, gfeats, qids, qids_sketch, gids, qimage_ids, qimage_ids_sketch, gimage_ids = self._compute_embedding(model)

        qfeats_text = F.normalize(qfeats_text, p=2, dim=1) # text features
        qfeats_sketch = F.normalize(qfeats_sketch, p=2, dim=1) # sketch features
        qfeats_text_sketch = F.normalize(qfeats_text_sketch, p=2, dim=1) # sketch+text features

        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity_text_rgb = qfeats_text @ gfeats.t()
        similarity_sketch_rgb = qfeats_sketch @ gfeats.t()
        similarity_textsketch_rgb = qfeats_text_sketch @ gfeats.t()
        
        #original gallery set for text-to-rgb retrieval
        t2i_cmc, t2i_mAP, t2i_mINP = eval_func(-similarity_text_rgb.detach().cpu().numpy() , qids.numpy(), gids.numpy(), qimage_ids.numpy(), gimage_ids.numpy(), set=0, max_rank=10)
        
        # remove the rgb images that used for generated sketches from gallery set
        t2i_cmc0, t2i_mAP0, t2i_mINP0 = eval_func(-similarity_text_rgb.detach().cpu().numpy() , qids.numpy(), gids.numpy(), qimage_ids.numpy(), gimage_ids.numpy(), set=2, max_rank=10)
        
        t2i_cmc1, t2i_mAP1, t2i_mINP1 = eval_func(-similarity_sketch_rgb.detach().cpu().numpy() , qids_sketch.numpy(), gids.numpy(), qimage_ids_sketch.numpy(), gimage_ids.numpy(), set=2, max_rank=10)
        t2i_cmc2, t2i_mAP2, t2i_mINP2 = eval_func(-similarity_textsketch_rgb.detach().cpu().numpy() , qids.numpy(), gids.numpy(), qimage_ids.numpy(), gimage_ids.numpy(), set=2, max_rank=10)

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i-text_RGB_original', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        table.add_row(['t2i-text_RGB', t2i_cmc0[0], t2i_cmc0[4], t2i_cmc0[9], t2i_mAP0, t2i_mINP0])
        table.add_row(['t2i-sketch_RGB', t2i_cmc1[0], t2i_cmc1[4], t2i_cmc1[9], t2i_mAP1, t2i_mINP1])
        table.add_row(['t2i-textsketch_RGB', t2i_cmc2[0], t2i_cmc2[4], t2i_cmc2[9], t2i_mAP2, t2i_mINP2])
        # table.add_row(['t2i-text_RGB', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mAP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, _ = rank(similarity=similarity_text_rgb.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP = i2t_cmc.cpu().numpy(), i2t_mAP.cpu().numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP])

        table.float_format = '.4'
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0], t2i_cmc1[0], t2i_cmc2[0]
    

    # def eval_by_proj(self, model, i2t_metric=False):

    #     qfeats, gfeats, qids, gids = self._compute_embedding(model)

    #     # qfeats_norm = F.normalize(qfeats, p=2, dim=1) # text features
    #     gfeats_norm = F.normalize(gfeats, p=2, dim=1) # image features

    #     similarity = qfeats @ gfeats_norm.t()

    #     t2i_cmc, t2i_mAP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    #     t2i_cmc, t2i_mAP = t2i_cmc.cpu().numpy(), t2i_mAP.cpu().numpy()
    #     table = PrettyTable(["task", "R1", "R5", "R10", "mAP"])
    #     table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP])

    #     if i2t_metric:
    #         i2t_cmc, i2t_mAP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
    #         i2t_cmc, i2t_mAP = i2t_cmc.cpu().numpy(), i2t_mAP.cpu().numpy()
    #         table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP])
    #     table.float_format = '.4'
    #     self.logger.info('\n' + str(table))
        
    #     return t2i_cmc[0]

