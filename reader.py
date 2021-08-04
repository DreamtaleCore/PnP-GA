import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import pathlib
import random


def gazeto2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])


# def crop(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     fa = np.where(img_gray > 0)  # face area
#     face_img = img[fa[0].min():fa[0].max() + 1, fa[1].min():fa[1].max() + 1]
#     return face_img


class loader(Dataset):
    def __init__(self, path, root, pic_num, header=True):
        self.lines = []
        self.pic_num = pic_num
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    if header: line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                self.lines = f.readlines()
                if header: self.lines.pop(0)
        if self.pic_num >= 0:
            self.lines = self.lines[:self.pic_num]
        self.root = pathlib.Path(root)

    def __len__(self):
        # if self.pic_num < 0:
        return len(self.lines)
        # return self.pic_num

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")
        # print(line)

        name = line[0].split('/')[0]
        gaze2d = line[1]
        head2d = line[2]
        # lefteye = line[1]
        # righteye = line[2]
        face = line[0]

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        headpose = np.array(head2d.split(",")).astype("float")
        headpose = torch.from_numpy(headpose).type(torch.FloatTensor)

        # rimg = cv2.imread(os.path.join(self.root, righteye))/255.0
        # rimg = rimg.transpose(2, 0, 1)

        # limg = cv2.imread(os.path.join(self.root, lefteye))/255.0
        # limg = limg.transpose(2, 0, 1)

        # print(self.root/name/ face)
        fimg = cv2.imread(str(self.root / face))
        # fimg=crop(fimg)
        # print(fimg.shape)
        fimg = cv2.resize(fimg, (448, 448)) / 255.0
        fimg = fimg.transpose(2, 0, 1)

        img = {"face": torch.from_numpy(fimg).type(torch.FloatTensor),
               "head_pose": headpose,
               "name": name}

        # img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
        #        "right":torch.from_numpy(rimg).type(torch.FloatTensor),
        #        "face":torch.from_numpy(fimg).type(torch.FloatTensor),
        #        "head_pose":headpose,
        #        "name":name}

        return img, label


def txtload(labelpath, imagepath, batch_size, pic_num=-1, shuffle=True, num_workers=0, header=True):
    # print(labelpath,imagepath)
    dataset = loader(labelpath, imagepath, pic_num, header)
    print(f"[Read Data]: Total num: {len(dataset)}")
    # print(f"[Read Data]: Label path: {labelpath}")
    # print(dataset.lines[:10])
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # seed_everything(1)
    path = '/home/lrc/gaze_datasets/EyeDiap/Label/p1.label'
    d = txtload(path, '/home/lrc/gaze_datasets/EyeDiap/Image', batch_size=32, pic_num=5,
                shuffle=True, num_workers=4, header=True)
    print(len(d))
    for i, (img, label) in enumerate(d):
        print(i, label)
