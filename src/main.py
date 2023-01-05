import os
import shutil
import tarfile
import urllib.request
from random import randint

import cv2
import numpy as np
import sklearn.metrics
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
path = 'VOC'


def get_archive(path, url):
    try:
        os.mkdir(path)
    except:
        path = path

    filename = 'devkit'
    urllib.request.urlretrieve(url, f"{path}/{filename}.tar")


get_archive(path, url)


def extract(path):
    tar_file = tarfile.open(f"{path}/devkit.tar")
    tar_file.extractall('./')
    tar_file.close()
    shutil.rmtree(path)


extract(path)

"""Various RGB palettes for coloring segmentation labels."""
VOC_CLASSES = [
    "background",
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
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class VocDataset(Dataset):
    def __init__(self, dir, color_map):
        self.root = os.path.join(dir, 'VOCdevkit/VOC2007')
        self.target_dir = os.path.join(self.root, 'SegmentationClass')
        self.images_dir = os.path.join(self.root, 'JPEGImages')
        file_list = os.path.join(self.root, 'ImageSets/Segmentation/trainval.txt')
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
        self.color_map = color_map

    def convert_to_segmentation_mask(self, mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)

        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        label_path = os.path.join(self.target_dir, f"{image_id}.png")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = torch.tensor(image).float()
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, (256, 256))
        label = self.convert_to_segmentation_mask(label)
        label = torch.tensor(label).float()
        return image, label

    def __len__(self):
        return len(self.files)


class FCN32(torch.nn.Module):
    def __init__(self, n_classes, pretrained_model):
        super(FCN32, self).__init__()
        self.pretrained_model = pretrained_model
        self.encoder = torch.nn.Sequential(*list(pretrained_model.features.children()))
        self.encoder_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(512, 4096, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Conv2d(4096, 4096, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(4096, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, n_classes, kernel_size=1),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.encoder(x)
        output = self.encoder_classifier(output)
        output = self.decoder(output)
        return output


def metrics(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = torch.argmax(y_true, dim=1)
    iou = sklearn.metrics.jaccard_score(y_true.cpu().flatten(), y_pred.cpu().flatten(), average='weighted')
    return iou


def getVal(x, y, loss_f, model):
    x, y = x.to(device), y.to(device)
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    y_pred = model(x)
    loss = loss_f(y_pred, y)
    return x, y, y_pred, loss


def train(model, optim, loss_f, epochs, scheduler, path_for_models):
    global device
    try:
        os.mkdir(path_for_models)
    except:
        path_for_models = path_for_models
    min_iou = 0.3
    for epoch in (range(epochs)):
        for (X_train, y_train) in train_loader:
            X_train, y_train, y_pred, loss = getVal(X_train, y_train, loss_f, model)
            optim.zero_grad()
            loss.backward()
            optim.step()
        ious = []
        val_losses = []
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(val_loader):
                X_test, y_test, y_val, val_loss = getVal(X_test, y_test, loss_f, model)
                val_losses.append(val_loss)
                iou_ = metrics(y_val, y_test)
                ious.append(iou_)
            ious = torch.tensor(ious)
            val_losses = torch.tensor(val_losses)
            scheduler.step(val_losses.mean())
            if ious.mean() > min_iou:
                min_iou = ious.mean()
                torch.save(model.state_dict(), f"{path_for_models}/fc32model.pth")
        print(f"Epoch: {epoch + 1}, Accuracy: {round(ious.mean() * 100, 3)}")


def getSegments(model, image):
    input_image = image.copy()
    input_image = torch.tensor(input_image).float()
    input_image = input_image.permute(2, 0, 1)
    input_image = input_image[None, :, :, :]
    input_image = input_image.to(device)

    with torch.no_grad():
        output_image = model(input_image)
    output_image = output_image[0].permute(1, 2, 0).cpu().numpy()
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            ind = np.argmax(output_image[i, j, :])
            output_image[i, j, ind] = 1
    output_image[output_image < 1] = 0
    segmap = np.ones((output_image.shape[0], output_image.shape[1], 3)) * 255
    for i in range(output_image.shape[2]):
        segmap[output_image[:, :, i] == 1, :] = VOC_COLORMAP[i]
    bg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = cv2.resize(segmap, (bg.shape[1], bg.shape[0]))
    alpha = 0.45
    beta = 1 - alpha
    dst = np.uint8(alpha * (bg) + beta * (overlay))
    return dst


dataset = VocDataset('/content', VOC_COLORMAP)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), round(len(dataset) * 0.1) + 1])
train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=10, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

pretrained_net = torchvision.models.vgg19(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FCN32(n_classes=21, pretrained_model=pretrained_net)
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

num_epochs = 10000

train(model, optimizer, criterion, num_epochs, scheduler, 'models')

torch.save(model, 'fcn32.pt')

model.eval().to(device)

model = torch.load('fcn32.pt')
model.eval().to(device)
image, _ = dataset.__getitem__(randint(0, dataset.__len__()))
image = np.asarray(image)

seg = getSegments(model, image)
cv2.imshow("Segmented", seg)
cv2.waitKey(0)
cv2.destroyAllWindows()
