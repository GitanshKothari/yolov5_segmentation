import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
import random
from utils.augmentations import letterbox, augment_hsv, random_perspective, cutout 
from utils.general import xyxy2xywh

class BDDDataset(Dataset):

    def __init__(self, cfg, mode, transform=None):
        
        assert mode in ["train", "val", "test"], "mode must be 'train', 'val' or 'test'"

        self.mode = mode
        self.cfg = cfg
        self.transform = transform if transform else T.Compose([T.ToTensor()])
        self.is_train = mode == "train"
        root_dir = Path(cfg['path'])
        self.img_dir = root_dir / "images" / mode
        self.det_dir = root_dir / "labels" / "det" / mode
        self.seg_dir = root_dir / "labels" / "seg" / mode
        
        self.db = self._build_database()
        self.resized_shape = 640
    
    def _build_database(self):

        print('Loading ' + self.mode + ' dataset...')
        gt_db = []
        self.labels = []
        for img_path in tqdm(self.img_dir.iterdir()):
            seg_path = self.seg_dir / img_path.name
            det_path = self.det_dir / img_path.name.replace('.jpg', '.txt')

            with open(det_path, 'r') as f:
                labels = f.readlines()

            gt = np.zeros((len(labels), 5))
            for i, label in enumerate(labels):

                label = [float(x) for x in label.split()]

                assert len(label) == 5, "Invalid label at " + det_path

                gt[i][0] = int(label[0])
                gt[i][1:] = label[1:]
            self.labels.append(gt)
            data = {
                'image': img_path,
                'label': gt,
                'mask': seg_path,
            }

            gt_db.append(data)
        print('Finished loading ' + self.mode + ' dataset.')
        return gt_db

    
    def __len__(self,):
        return len(self.db)
    
    def __getitem__(self, idx):
        
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg_label = cv2.imread(data["mask"], 0)
        if isinstance(self.resized_shape, list):
            self.resized_shape = max(self.resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = self.resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_NEAREST_EXACT)
        h, w = img.shape[:2]
        
        img, ratio, pad = letterbox(img, self.resized_shape, auto=True, scaleup=self.is_train)
        seg_label, _, _ = letterbox(seg_label, self.resized_shape, color = (0, 0, 0), auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        
        det_label = data["label"]
        labels=[]
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
            
        if self.is_train:
            
            (img, seg_label), labels = random_perspective(
                combination=(img, seg_label),
                targets=labels,
            )

            augment_hsv(img)

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                seg_label = np.fliplr(seg_label)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.filpud(seg_label)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]
        
        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        
        img = np.ascontiguousarray(img)


        seg_background = np.where(seg_label == 0, 1, 0)
        seg_drivable = np.where(seg_label >= 0, 1, 0)
        
        seg_label = np.stack([seg_background, seg_drivable], axis=2)
        seg_label = self.transform(seg_label)

        target = [labels_out, seg_label]
        
        img = self.transform(img)

        return img, target, data["image"], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, label_seg = [], []
        for i, l in enumerate(label):
            l_det, l_seg= l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_seg, 0)], paths, shapes
    


if __name__ == "__main__":
    import yaml

    cfg_path = r'data\bdd.yaml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset = BDDDataset(cfg, 'train')
    print(len(dataset))
    print(dataset.db[0])

