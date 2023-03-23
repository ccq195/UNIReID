import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class RSTPReid(BaseDataset):
    """
    ICFG-PEDES

    Reference:
    Semantically Self-Aligned Network for Text-to-Image Part-aware Person Re-identification arXiv 2107

    URL: https://github.com/NjtechCVLab/RSTPReid-Dataset

    Dataset statistics:
    # identities: 4101, 3701 + 200 + 200
    # Each person has 5 corresponding images taken by different cameras with complex both indoor and outdoor scene, Each image is annotated with 2 textual descriptions.
    # images: 18505  (train) + 1000 (val) + 1000 (text)
    # cameras: 15
    """
    dataset_dir = 'RSTPReid'

    def __init__(self, root='', nlp_aug=False, verbose=True):
        super(RSTPReid, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')
        self.simg_dir = op.join(self.dataset_dir, 'imgs-sketch/')
        if nlp_aug:
            ## not implement yet
            # self.anno_path = op.join(self.dataset_dir, 'nlp_aug.json')
            raise FileNotFoundError
        else:
            self.anno_path = op.join(self.dataset_dir, 'data_captions.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> RSTPReid Images and Captions are loaded")
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
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['img_path'])
                captions = anno['captions'] # caption list
                simg_path = op.join(self.simg_dir, anno['img_path'])

                for caption in captions:
                    dataset.append((pid, image_id, img_path, simg_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            image_id = 0
            simage_ids = []
            simage_pids = []
            simg_paths = []
            image_ids = []

            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['img_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                image_ids.append(image_id)
                simg_path =  op.join(self.simg_dir, anno['img_path']) 

                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
                    simg_paths.append(simg_path)
                    simage_ids.append(image_id)

                simage_pids.append(pid)
                image_id += 1

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
