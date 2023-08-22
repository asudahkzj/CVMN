import csv
import os
import random

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from bert_embedding import BertEmbedding
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from skimage import io
from skimage.segmentation import slic
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as F

from .utils import resize_and_pad, resize
import datasets.transforms as T
import clip


class A2DSubset(Dataset):
    def __init__(self, image_set, args, num_frames):
        self.word2vec = KeyedVectors.load_word2vec_format(args['vocab_path'], binary=True)
        self.args = args
        self.col_path = os.path.join(args['annotation_path'], 'col')  # rgb
        self.mat_path = os.path.join(args['annotation_path'], 'mat')  # matrix
        self.max_num_words = args['max_num_words']

        self.bert_embedding = BertEmbedding()
        self.clip_preprocess = clip.load("RN50")[1]
        self._read_video_info()
        self._read_dataset_samples()
        
        self.num_frames = num_frames
        self.videos = self.train_videos
        self.samples = self.train_samples
        self._transforms = make_coco_transforms(image_set)

        self.is_full = np.zeros(len(self.samples), dtype=np.int64)
        np.random.seed(88)
        self.full_videos = list(self.train_videos.keys())
        np.random.shuffle(self.full_videos)
        self.full_videos = self.full_videos[:int(0.5 * len(self.full_videos))]
        # self.train = train

        self.extract_query = {}
        for i in range(len(self.train_samples)):
            numbers = random.sample(range(0, len(self.train_query)), 10)
            self.extract_query[i] = numbers


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_id, instance_id, frame_idx, query = self.samples[index]
        query = query.lower()
        frame_idx = int(frame_idx)
        
        h5_path = os.path.join('data/a2d/a2d_annotation_with_instances', video_id,
                               '%05d.h5' % (frame_idx + 1))
        if not os.path.exists(h5_path):
            h5_path = os.path.join('data/a2d/a2d_annotation_with_instances', video_id,
                                   '%05d.h5' % (24 + 1))
        frame_path = os.path.join('data/a2d/Release/pngs320H', video_id)

        frames = list(map(lambda x: os.path.join(frame_path, x),
                          sorted(os.listdir(frame_path))))
        # print(len(frames), self.videos['num_frames'])

        assert len(frames) == self.videos[video_id]['num_frames']

        all_frames = []
        mid_frame = (self.num_frames-1)//2
        for i in range(self.num_frames):
            all_frames.append(frame_idx-mid_frame+i)
        # all_frames = [i for i in range(frame_idx - 4 * step, frame_idx + 4 * step, step)]
        for i in range(len(all_frames)):
            if all_frames[i] < 0:
                all_frames[i] = 0
            elif all_frames[i] >= len(frames):
                all_frames[i] = len(frames) - 1
        all_frames = np.asarray(frames)[all_frames]

        img = []
        for i in all_frames:
            img.append(Image.open(i).convert('RGB'))

        img_clip = torch.stack([self.clip_preprocess(im) for im in img])
        # img_clip, text_clip = None, None

        expressions = []
        # expressions.append(np.zeros((7, 768)))
        # text_clip = None
        expressions.append(query)
        numbers = self.extract_query[index]
        for i in range(10):
            query = self.train_query[numbers[i]]
            expressions.append(query)
        text_clip = clip.tokenize(expressions)
        results = self.bert_embedding(expressions)
        expressions = [np.asarray(result[1]) for result in results]

        # fine-grained mask
        with h5py.File(h5_path, mode='r') as fp:
            instance = np.asarray(fp['instance'])
            all_masks = np.asarray(fp['reMask'])
            if len(all_masks.shape) == 3 and instance.shape[0] != all_masks.shape[0]:
                print(video_id, frame_idx + 1, instance.shape, all_masks.shape)

            all_boxes = np.asarray(fp['reBBox']).transpose([1, 0])  # [w_min, h_min, w_max, h_max]
            all_ids = np.asarray(fp['id'])
            # if video_id == 'EadxBPmQvtg' and frame_idx == 24:
            #     instance = instance[:-1]
            assert len(all_masks.shape) == 2 or len(all_masks.shape) == 3
            if len(all_masks.shape) == 2:
                mask = all_masks[np.newaxis]
                class_id = int(all_ids[0][0])
                coarse_gt_box = all_boxes[0]
            else:
                instance_id = int(instance_id)
                idx = np.where(instance == instance_id)[0][0]

                mask = all_masks[idx]
                coarse_gt_box = all_boxes[idx]
                class_id = int(all_ids[0][idx])
                mask = mask[np.newaxis]

            #
            # print(class_id, class_name)

            assert len(mask.shape) == 3
            assert mask.shape[0] > 0

            fine_gt_mask = np.transpose(np.asarray(mask), (0, 2, 1))[0]

        pos = np.where(fine_gt_mask==1)
        x1, y1, x2, y2 = np.min(pos[0]), np.min(pos[1]), np.max(pos[0]), np.max(pos[1])
        coarse_gt_box1 = np.array([y1, x1, y2, x2])
        area = np.sum(fine_gt_mask)
        w, h = img[0].size
        target = {}
        target['boxes'] = torch.from_numpy(coarse_gt_box).float().unsqueeze(0)
        target['labels'] = torch.from_numpy(np.array([class_id]))
        target['masks'] = torch.from_numpy(fine_gt_mask).unsqueeze(0)
        target['image_id'] = torch.tensor([index])
        target['valid'] = torch.tensor([1])
        target['area'] = torch.tensor([int(area)])
        target['iscrowd'] = torch.tensor([0])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return torch.cat(img,dim=0), expressions, target, (img_clip, text_clip)         

    def _read_video_info(self):
        self.train_videos, self.test_videos = {}, {}
        with open(self.args['videoset_path'], newline='') as fp:
            reader = csv.reader(fp, delimiter=',')
            for row in reader:
                frame_idx = list(map(lambda x: int(x[:-4]) - 1,
                                     os.listdir(os.path.join(self.col_path, row[0]))))
                frame_idx = sorted(frame_idx)
                # print(frame_idx)
                # exit(0)
                video_info = {
                    'label': int(row[1]),
                    'timestamps': [row[2], row[3]],
                    'size': [int(row[4]), int(row[5])],  # [height, width]
                    'num_frames': int(row[6]),
                    'num_annotations': int(row[7]),
                    'frame_idx': frame_idx,
                }
                if int(row[8]) == 0:
                    self.train_videos[row[0]] = video_info
                else:
                    self.test_videos[row[0]] = video_info

    def _read_dataset_samples(self):
        self.train_samples, self.test_samples = [], []
        self.train_videos_set = set()
        self.test_videos_set = set()
        self.all_query = set()
        self.train_query = []
        with open(self.args['sample_path'], newline='') as fp:
            reader = csv.DictReader(fp)
            from collections import defaultdict
            video2frame = defaultdict(list)
            rows = []
            query = ''
            for row in reader:
                rows.append(row)
                video2frame[(row['video_id'], row['query'])].append(row['frame_idx'])
            for row in rows:
                if row['video_id'] in self.train_videos:
                    self.train_samples.append([row['video_id'], row['instance_id'], row['frame_idx'], row['query']])
                    self.train_videos_set.add(row['video_id'])
                    cur_query = row['query'].lower()
                    if query != cur_query:
                        query = cur_query
                        self.train_query.append(cur_query)
                else:
                    l = video2frame[(row['video_id'], row['query'])]
                    # if l[len(l) >> 1] != row['frame_idx']:
                    #     continue
                    self.test_samples.append([row['video_id'], row['instance_id'], row['frame_idx'], row['query']])
                    self.test_videos_set.add(row['video_id'])
                    # self.test_samples.append([row['video_id'], row['instance_id'], 36, row['query']])
                    # self.test_samples.append([row['video_id'], row['instance_id'], 61, row['query']])
                self.all_query.add((row['video_id'], row['query']))
        print('number of sentences: {}'.format(len(self.all_query)))
        print('videos for training: {}, videos for testing: {}'.format(len(self.train_videos_set),
                                                                       len(self.test_videos_set)))
        print(
            'samples for training: {}, samples for testing: {}'.format(len(self.train_samples), len(self.test_samples)))
        # exit(0)


class A2DSlicRGB:
    def __init__(self, image_set, args):
        self.word2vec = KeyedVectors.load_word2vec_format(args['vocab_path'], binary=True)
        # self.word2vec.save()

        self.args = args
        self.col_path = os.path.join(args['annotation_path'], 'col')  # rgb
        self.mat_path = os.path.join(args['annotation_path'], 'mat')  # matrix
        self.max_num_words = args['max_num_words']
        self.resolution = [8, 16, 32, 64, 128, 256]
        self.resolution = [10, 20, 40, 80, 160, 320]
        self.n_segments = {
            self.resolution[0]: 16,
            self.resolution[1]: 32,
            self.resolution[2]: 32,
            self.resolution[3]: 32,
            self.resolution[4]: 32,
            self.resolution[5]: 32,
        }
        import pickle
        with open('cnt.pkl', 'rb') as fp:
            self.id2idx = pickle.load(fp)

        self.bert_embedding = BertEmbedding()
        self._read_video_info()
        self._read_dataset_samples()
        # self.train_set_ = A2DSubset(self.train_videos, self.train_samples, self.word2vec, args,
        #                             transforms=transforms.Compose([
        #                                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        #                                 transforms.RandomHorizontalFlip(),
        #                                 lambda x: np.asarray(x),
        #                             ]), train=True)
        self.train_set_ = A2DSubset(self.train_videos, self.train_samples, self.word2vec, self.bert_embedding, self.id2idx, args,
                                    transforms=make_coco_transforms(image_set), train=True)
        self.test_set_ = A2DSubset(self.test_videos, self.test_samples, self.word2vec, self.bert_embedding, self.id2idx, args)

    def _read_video_info(self):
        self.train_videos, self.test_videos = {}, {}
        with open(self.args['videoset_path'], newline='') as fp:
            reader = csv.reader(fp, delimiter=',')
            for row in reader:
                frame_idx = list(map(lambda x: int(x[:-4]) - 1,
                                     os.listdir(os.path.join(self.col_path, row[0]))))
                frame_idx = sorted(frame_idx)
                # print(frame_idx)
                # exit(0)
                video_info = {
                    'label': int(row[1]),
                    'timestamps': [row[2], row[3]],
                    'size': [int(row[4]), int(row[5])],  # [height, width]
                    'num_frames': int(row[6]),
                    'num_annotations': int(row[7]),
                    'frame_idx': frame_idx,
                }
                if int(row[8]) == 0:
                    self.train_videos[row[0]] = video_info
                else:
                    self.test_videos[row[0]] = video_info

    def _read_dataset_samples(self):
        self.train_samples, self.test_samples = [], []
        self.train_videos_set = set()
        self.test_videos_set = set()
        self.all_query = set()
        with open(self.args['sample_path'], newline='') as fp:
            reader = csv.DictReader(fp)
            from collections import defaultdict
            video2frame = defaultdict(list)
            rows = []
            for row in reader:
                rows.append(row)
                video2frame[(row['video_id'], row['query'])].append(row['frame_idx'])
            for row in rows:
                if row['video_id'] in self.train_videos:
                    self.train_samples.append([row['video_id'], row['instance_id'], row['frame_idx'], row['query']])
                    self.train_videos_set.add(row['video_id'])
                else:
                    l = video2frame[(row['video_id'], row['query'])]
                    # if l[len(l) >> 1] != row['frame_idx']:
                    #     continue
                    self.test_samples.append([row['video_id'], row['instance_id'], row['frame_idx'], row['query']])
                    self.test_videos_set.add(row['video_id'])
                    # self.test_samples.append([row['video_id'], row['instance_id'], 36, row['query']])
                    # self.test_samples.append([row['video_id'], row['instance_id'], 61, row['query']])
                self.all_query.add((row['video_id'], row['query']))
        print('number of sentences: {}'.format(len(self.all_query)))
        print('videos for training: {}, videos for testing: {}'.format(len(self.train_videos_set),
                                                                       len(self.test_videos_set)))
        print(
            'samples for training: {}, samples for testing: {}'.format(len(self.train_samples), len(self.test_samples)))
        # exit(0)

    @property
    def train_set(self):
        return self.train_set_

    @property
    def test_set(self):
        return self.test_set_


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=800),
            T.PhotometricDistort(),
            T.Compose([
                     T.RandomResize([400, 500, 600]),
                     T.RandomSizeCrop(384, 600),
                     # To suit the GPU memory the scale might be different
                     T.RandomResize([300], max_size=540),#for r50
                     #T.RandomResize([280], max_size=504),#for r101
            ]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    paths = {
        "videoset_path": "data/a2d/Release/videoset.csv",
        "annotation_path": "data/a2d/Release/Annotations",
        "sample_path": "data/a2d/a2d_annotation_info.txt",
    }
    dataset = A2DSubset(image_set, paths, num_frames = args.num_frames)
    return dataset


if __name__ == '__main__':
    args = {
        "videoset_path": "data/a2d/Release/videoset.csv",
        "annotation_path": "data/a2d/Release/Annotations",
        "sample_path": "data/a2d/a2d_annotation_info.txt",
    }
    dataset = A2DSlicRGB(args)
    dataset.train_set[66]
    # exit(0)
    import pickle
    loader = DataLoader(dataset.train_set, batch_size=32, shuffle=True, num_workers=1,
                        pin_memory=True, collate_fn=dataset.collate_fn)
    id2idx = {}
    id2idx[0] = 0
    for batch in loader:
        print(batch['net_input']['query_idx'])
        for i in batch['net_input']['query_idx']:
            for j in i:
                if int(j) not in id2idx:
                    id2idx[int(j)] = len(id2idx)
        print(len(id2idx))

        # for k, v in batch['net_input'].items():
        #     print(k, v.size())

    with open('cnt.pkl', 'wb') as fp:
        pickle.dump(id2idx, fp)