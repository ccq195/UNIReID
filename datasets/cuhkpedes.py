from json.encoder import py_encode_basestring
import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset
import numpy as np
import pdb

class CUHKPEDES(BaseDataset):
    """
    CUHK-PEDES

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    Dataset statistics:
    ### identities: 13003
    ### images: 40206,  (train)  (test)  (val)
    ### captions: 
    ### 9 images have more than 2 captions
    ### 4 identity have only one image

    annotation format: 
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = 'CUHK-PEDES'

    def __init__(self, root='', nlp_aug=False, verbose=True):
        super(CUHKPEDES, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')
        self.simg_dir = op.join(self.dataset_dir, 'imgs-sketch2/')
        if nlp_aug:
            self.anno_path = op.join(self.dataset_dir, 'nlp_aug.json') 
        else:
            self.anno_path = op.join(self.dataset_dir, 'reid_raw.json') 
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0

            for anno in annos:
                pid = int(anno['id']) - 1 # make pid begin from 0
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                # if pid not in pid_container:
                simg_path = op.join(self.simg_dir, anno['file_path'])
                pid_container.add(pid)

                for caption in captions:
                    dataset.append((pid, image_id, img_path, simg_path, caption))
                image_id += 1


            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"

            # for i in pid_container:
            #     indexs = [z[0] for z in list(enumerate(images_pid)) if z[1] == i]
            #     index = int(np.random.choice(indexs,1)) # 每个身份随机选择一张sektch  #,选择多张sketch呢，与caption对应张sektch？ numpy.random.choice(aaa, 5)
            #     simg_paths[i] = op.join(self.simg_dir, rgb_name[index])
            #     simage_ids[i] = image_ids[index]
            
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            simg_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            image_id = 0
            image_ids = []
            simage_ids = []
            simage_pids = []

            for anno in annos:
                pid = int(anno['id'])
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                image_ids.append(image_id)
           
                # if pid not in pid_container:
                simg_path =  op.join(self.simg_dir, anno['file_path']) 
                    #  simage_id = image_id
                    #  simage_pid = pid

                pid_container.add(pid)

                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)

                    simg_paths.append(simg_path)
                    simage_ids.append(image_id)
                simage_pids.append(pid)

                image_id += 1
                
            # for i in pid_container:
            #     indexs = [z[0] for z in list(enumerate(images_pid)) if z[1] == i]
            #     index = int(np.random.choice(indexs,1)) # 每个身份随机选择一张sektch  #,选择多张sketch呢，与caption对应张sektch？ numpy.random.choice(aaa, 5)
            #     simg_paths[i] = op.join(self.simg_dir, rgb_name[index])
            #     simage_ids[i] = image_ids[index]

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "image_ids": image_ids,
                "simage_pids": simage_pids,
                "simg_paths": simg_paths,
                "simage_ids": simage_ids,
                "caption_pids": caption_pids,
                "captions": captions 
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
