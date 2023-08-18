import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from utils import seed_everything

DATASET = 'PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 2
BATCH_SIZE  = 8
#IMAGE_SIZES  = [364,416,468]
NUM_CLASSES = 20
LEARNING_RATE  = 1e-5
WEIGHT_DECAY   = 1e-4
NUM_EPOCHS     = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
#S = [[IMAGE_SIZES[0] // 28, IMAGE_SIZES[0] // 14, IMAGE_SIZES[0] // 7],[IMAGE_SIZES[1] // 32, IMAGE_SIZES[1]  // 16, IMAGE_SIZES[1]  // 8],[IMAGE_SIZES[2] // 36, IMAGE_SIZES[2] // 18, IMAGE_SIZES[2] // 9]]
DIV = 32
IMAGE_SIZES = [320, 416, 608, 416, 416]
S = [[x//DIV, x//DIV*2, x//DIV*4] for x in IMAGE_SIZES]

PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = "/s13/PASCAL_VOC/images/"
LABEL_DIR = "/s13/PASCAL_VOC/labels/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

means = [0.485, 0.456, 0.406]

scale = 1.1
def custom_train_transform(IMAGE_SIZE):
    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
            A.PadIfNeeded(
                min_height=int(IMAGE_SIZE * scale),
                min_width=int(IMAGE_SIZE * scale),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Rotate(limit = 10, interpolation=1, border_mode=4),
            A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                    ),
                    # A.Affine(shear=15, p=0.5, mode="constant"),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
    )

    return(train_transforms)

def custom_test_transform(IMAGE_SIZE):
    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )

    return(test_transforms)

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]
