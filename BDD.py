import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

class BDDDataset(Dataset):

    def __init__(self, cfg, mode, transform=None):
        
        assert mode in ["train", "val", "test"], "mode must be 'train', 'val' or 'test'"

        self.mode = mode
        self.cfg = cfg
        self.transform = transform

        root_dir = Path(cfg['path'])
        self.img_dir = root_dir / "images" / mode
        self.det_dir = root_dir / "labels" / "det" / mode
        self.seg_dir = root_dir / "labels" / "seg" / mode

        self.db = self._build_database()

    
    def _build_database(self):

        print('Loading ' + self.mode + ' dataset...')
        gt_db = []
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


if __name__ == "__main__":
    import yaml

    cfg_path = r'data\bdd.yaml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset = BDDDataset(cfg, 'train')
    print(len(dataset))
    print(dataset.db[0])

