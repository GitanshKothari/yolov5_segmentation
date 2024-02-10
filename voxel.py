import fiftyone as fo
from pathlib import Path


name = "BDDDataset"
root_dir = Path(r"C:\Users\gitan\ML_Projects\yolov5_segmentation") / name
img_dir = root_dir / "images" / "train"
det_dir = root_dir / "labels" / "det" / "train"
seg_dir = root_dir / "labels" / "seg" / "train"

datasets = fo.list_datasets()
if name in datasets:
    fo.delete_dataset(name)

# Creating the dataset
dataset = fo.Dataset.from_images_dir(
    img_dir,
    name=name,
    persistent = True
)

detection_classes_reverse = {
    0: 'pedestrian',
    1: 'rider',
    2: 'car',
    3: 'truck',
    4: 'bus',
    5: 'train',
    6: 'motorcycle',
    7: 'bicycle',
    8: 'traffic light',
    9: 'traffic sign',
}

# Looping over all the samples, reading the detection from the json files and adding the segmentation masks
with fo.ProgressBar() as pb:
    for sample in pb(dataset):
        w, h = 1280, 720

        with open(det_dir / Path(sample.filepath).with_suffix('.txt').name) as f:
            labels = f.readlines()
        
        detections = []
        for label in labels:
            label = [float(x) for x in label.split()]
            label[1] = label[1] - label[3] / 2 
            label[2] = label[2] - label[4] / 2 
            detections.append(
                fo.Detection(
                    label=detection_classes_reverse[int(label[0])],
                    bounding_box=label[1:5],
                )
            )

        sample["detections"] = fo.Detections(detections=detections)

        segmentation_path = seg_dir / Path(sample.filepath).name
        sample["segmentation"] = fo.Segmentation(mask_path = str(segmentation_path))
        
        sample.save()


dataset = fo.load_dataset(name)
session = fo.launch_app(dataset, auto = False, desktop=True)
session.wait()