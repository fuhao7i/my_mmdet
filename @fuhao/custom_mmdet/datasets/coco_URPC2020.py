from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class coco_URPC2020(CocoDataset):
    CLASSES = ("holothurian", "echinus", "scallop", "starfish")
