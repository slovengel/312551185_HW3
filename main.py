import random
import os
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm.auto import tqdm
from PIL import Image
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.io as sio
from utils import encode_mask
from pycocotools import mask as mask_utils

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

CLASS_NAMES = ['background', 'class1', 'class2', 'class3', 'class4']
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCH = 20


class CellSegmentationDataset(Dataset):
    def __init__(self, transforms=None):
        self.root_dir = 'C:\\Users\\user\\Documents\\hw3-data-release\\train'
        self.transforms = transforms
        self.samples = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.samples[idx])
        image_path = os.path.join(sample_dir, 'image.tif')
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        masks = []
        boxes = []
        labels = []

        for i, class_name in enumerate(CLASS_NAMES[1:], start=1):
            class_mask_path = os.path.join(sample_dir, f'{class_name}.tif')
            if not os.path.exists(class_mask_path):
                continue

            class_mask = np.array(sio.imread(class_mask_path))

            instance_ids = np.unique(class_mask)
            instance_ids = instance_ids[instance_ids != 0]

            for inst_id in instance_ids:
                mask = (class_mask == inst_id).astype(np.uint8)
                masks.append(mask)

                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(i)

        if not boxes:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
            masks = [np.zeros_like(image[:, :, 0], dtype=np.uint8)]

        masks = np.array(masks)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image = Image.fromarray(image)
        if self.transforms:
            image = self.transforms(image)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
        }

        return image, target, self.samples[idx], idx


transform = Compose([
    ToTensor(),
])

dataset = CellSegmentationDataset(transforms=transform)
train_dataflow = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: list(zip(*x))
)

categories = [{
    "id": i,
    "name": name
} for i, name in enumerate(CLASS_NAMES) if i != 0]

coco = {
    "images": [],
    "annotations": [],
    "categories": categories,
}

ann_id = 1

for idx in tqdm(range(len(dataset))):
    _, target, file_name, _ = dataset[idx]

    img_path = os.path.join(dataset.root_dir, file_name, "image.tif")
    with Image.open(img_path) as img:
        width, height = img.size

    coco["images"].append({
        "id": idx,
        "file_name": f"{file_name}/image.tif",
        "width": width,
        "height": height
    })

    boxes = target['boxes']
    labels = target['labels']
    masks = target['masks']

    for box, label, mask in zip(boxes, labels, masks):
        box = box.tolist()
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y

        binary_mask = mask.numpy()
        rle = encode_mask(binary_mask)
        area = mask_utils.area(rle).item()

        coco["annotations"].append({
            "id": ann_id,
            "image_id": idx,
            "category_id": label.item(),
            "bbox": [x, y, w, h],
            "segmentation": rle,
            "area": area,
            "iscrowd": 0,
        })
        ann_id += 1

with open("ground_truth.json", "w") as f:
    json.dump(coco, f)

model = maskrcnn_resnet50_fpn_v2(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask,
    hidden_layer,
    NUM_CLASSES
)


def train(
    model: nn.Module,
    dataflow: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler,
):
    model.train()

    total_loss = 0
    count = 0
    for images, targets, file_names, image_ids in tqdm(
        dataflow,
        desc='train',
        leave=False
    ):
        images = list(img.cuda() for img in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        count += 1

    scheduler.step()

    avg_loss = total_loss / count
    return avg_loss


def evaluate(
    model: nn.Module,
    dataflow: DataLoader,
    gt_json_path: str,
    dt_json_path: str,
):

    results = []
    model.eval()

    count = 0

    with torch.no_grad():
        for images, targets, file_names, image_ids in tqdm(
            dataflow,
            desc='eval',
            leave=False
        ):
            count += 1

            images = list(img.cuda() for img in images)

            with torch.amp.autocast('cuda'):
                outputs = model(images)

            for output, image_id in zip(outputs, image_ids):
                image_id = int(image_id)

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                masks = output['masks'].cpu().numpy()
                masks = masks > 0.5

                for box, score, label, mask in zip(
                    boxes,
                    scores,
                    labels,
                    masks
                ):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1

                    binary_mask = mask[0].astype(np.uint8)
                    rle = encode_mask(binary_mask)

                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "segmentation": rle,
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(width),
                            float(height)
                        ],
                        "score": float(score)
                    })

    with open(dt_json_path, "w") as f:
        json.dump(results, f)

    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dt_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map = coco_eval.stats[1]  # mAP@0.50 (IoU=0.5)

    return map


optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=NUM_EPOCH,
    eta_min=1e-6
)
scaler = torch.amp.GradScaler()

root_dir = 'C:\\Users\\user\\Documents\\'

model.cuda()

for epoch in range(NUM_EPOCH):
    print("Epoch", epoch + 1, ":")

    train_loss = train(model, train_dataflow, optimizer, scheduler, scaler)
    torch.cuda.empty_cache()
    train_map = evaluate(
        model,
        train_dataflow,
        root_dir + "ground_truth.json",
        f"train_preds_{epoch+1}.json"
    )
    torch.cuda.empty_cache()

    print(
        f"Epoch {epoch + 1}:\t"
        f"Train Loss: {train_loss:.4f} "
        f"Train mAP: {train_map:.4f}"
    )

    torch.save(model.state_dict(), f"maskrcnn_{epoch+1}.pt")


class CellSegmentationDataset_test(Dataset):
    def __init__(self, transforms=None):
        self.root_dir = 'C:\\Users\\user\\Documents\\hw3-data-release\\'
        self.transforms = transforms

        with open(self.root_dir + 'test_image_name_to_ids.json', 'r') as f:
            image_name_to_ids = json.load(f)

        self.images = []
        for col in image_name_to_ids:
            self.images.append({
                'image_id': int(col["id"]),
                'file_name': col["file_name"],
                'size': [col["height"], col["width"]]
            })

    def __getitem__(self, idx):
        image_info = self.images[idx]
        img_path = os.path.join(
            self.root_dir + 'test_release',
            image_info['file_name']
        )
        image = Image.open(img_path).convert("RGB")
        image_id = image_info['image_id']
        if self.transforms:
            image = self.transforms(image)

        return image, image_id

    def __len__(self):
        return len(self.images)


test_dataset = CellSegmentationDataset_test(transforms=transform)
test_dataflow = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: list(zip(*x))
)


def inference(
    model: nn.Module,
    dataflow: DataLoader,
    dt_json_path: str,
):

    results = []
    count = 0
    model.eval()

    with torch.no_grad():
        for images, image_ids in tqdm(dataflow, desc='eval', leave=False):
            count += 1

            images = list(img.cuda() for img in images)

            outputs = model(images)

            for output, image_id in zip(outputs, image_ids):
                image_id = int(image_id)

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                masks = output['masks'].cpu().numpy()
                masks = masks > 0.5

                for box, score, label, mask in zip(
                    boxes,
                    scores,
                    labels,
                    masks
                ):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1

                    binary_mask = mask[0].astype(np.uint8)
                    rle = encode_mask(binary_mask)

                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "segmentation": rle,
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(width),
                            float(height)
                        ],
                        "score": float(score)
                    })

    with open(dt_json_path, "w") as f:
        json.dump(results, f)


model.load_state_dict(torch.load('maskrcnn_10.pt', weights_only=True))
model.cuda()
inference(model, test_dataflow, "test_preds_10.json")
