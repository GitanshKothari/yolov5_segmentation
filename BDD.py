from pathlib import Path
from torch.utils.data import Dataset

class BDDDataset(Dataset):

    def __init__(self, cfg, mode, transform=None):
        
        assert mode in ["train", "val", "test"], "mode must be 'train', 'val' or 'test'"

        self.mode = mode
        self.cfg = cfg
        self.transform = transform

        root_dir = Path(cfg['path'])
        img_dir = root_dir / "images" / mode
        det_dir = root_dir / "labels" / "det" / mode
        seg_dir = root_dir / "labels" / "seg" / mode

        self.img_list = list(img_dir.glob("*.jpg"))
        self.db = []


