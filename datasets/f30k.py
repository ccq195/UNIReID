import os.path as op
from typing import List
from utils.iotools import read_json
from .bases import BaseDataset


class F30K(BaseDataset):
    """
    Flickr30K

    Reference:
    From image descriptions to visual denotations:
    New similarity metrics for semantic inference over event descriptions

    URL: https://aclanthology.org/Q14-1006/

    Dataset statistics:
    The Flickr30k dataset contains 31,000 images collected from Flickr, together with 5 reference sentences provided by human annotators.

    annotation format: 
    {
        'images': list[{
            'sentids': [10, 11, 12, 13, 14], 
            'imgid': 2, 
            'sentences': [{
                'tokens': ['a', 'child', 'in', 'a', 'pink', 'dress'], 
                'raw': 'A child in a pink dress',
                'imgid': 2 , 
                'sentid': 10
            }, ...], 
            'split': 'train', 
            'filename': '1000268201.jpg'
        }, ...],
        'dataset': 'flickr30k',
    }
    """
    dataset_dir = 'F30K'

    def __init__(self, root='', nlp_aug=False, verbose=True):
        super(F30K, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'flickr30k-images/')
        if nlp_aug:
            self.anno_path = op.join(self.dataset_dir, 'nlp_aug.json')
        else:
            self.anno_path = op.join(self.dataset_dir, 'karpathy/dataset_flickr30k.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> F30K Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)['images']
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
            for anno in annos:
                img_path = op.join(self.img_dir, anno['filename'])
                img_id = anno['imgid']
                sentences = anno['sentences'] # caption list
                for sentence in sentences:
                    assert img_id == sentence['imgid']
                    caption = sentence['raw']
                    dataset.append((img_id, -1,  img_path, caption))

            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            img_ids = []
            caption_pids = []
            for anno in annos:

                img_path = op.join(self.img_dir, anno['filename'])
                img_paths.append(img_path)
                img_id = anno['imgid']
                img_ids.append(img_id)

                sentences = anno['sentences'] # caption list
                for sentence in sentences:
                    captions.append(sentence['raw'])
                    caption_pids.append(sentence['imgid'])
            dataset = {
                "image_pids": img_ids,
                "img_paths": img_paths,
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

