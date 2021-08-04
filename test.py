import model as model
import reader as reader
import numpy as np
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy


def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


# def apply_dropout(m):
#     if type(m) == nn.Dropout:
#         m.train()


def angular(gaze, label):
    gaze = gazeto3d(gaze)
    label = gazeto3d(label)
    total = np.sum(gaze * label)
    return np.arccos(min(total / (np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999)) * 180 / np.pi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--i', type=int, default=0,
                        help="i represents the i-th folder used as the test set")
    parser.add_argument('--savepath', type=str, default=None, help="path to save models")
    parser.add_argument('--gpu', type=int, default=0, help="gpu")
    parser.add_argument('--target', type=str, default='mpii', help="target dataset, mpii/edp")
    parser.add_argument('--p', type=int, default=0, help="person")
    parser.add_argument('--model_name', type=str, default='GA', help="model name")
    args = parser.parse_args()

    config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    config = config["test"]

    loc = f'cuda:{args.gpu}'

    imagepath = config[args.target]["image"]
    labelpath = config[args.target]["label"]
    modelname = args.model_name

    loadpath = os.path.join(args.savepath)
    device = torch.device(loc if torch.cuda.is_available() else "cpu")

    folder = os.listdir(labelpath)
    folder.sort()
    i = args.i
    model_dir = args.p

    if i in range(len(folder)):
        tests = folder[i]
    else:
        tests = f'cross-{args.target}'
    print(f"Test Set: {tests}")

    savepath = os.path.join(loadpath, f"checkpoint/{folder[model_dir]}")
    # savepath = os.path.join(loadpath, f"")

    if not os.path.exists(os.path.join(loadpath, f"evaluation_{folder[model_dir]}/{tests}")):
        os.makedirs(os.path.join(loadpath, f"evaluation_{folder[model_dir]}/{tests}"))

    print("Read data")
    if i in range(len(folder)):
        dataset = reader.txtload(os.path.join(labelpath, tests), imagepath, 80, num_workers=4, header=True,
                                 shuffle=False)
    else:
        dataset = reader.txtload([os.path.join(labelpath, test) for test in folder], imagepath, 80, num_workers=4,
                                 header=True, shuffle=False)

    begin = config["load"]["begin_step"]
    end = config["load"]["end_step"]
    step = config["load"]["steps"]

    for saveiter in range(begin, end + step, step):
        print("Model building")
        net = model.GazeRes18()
        statedict = torch.load(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"), map_location=loc)
        statedict = {k.replace('module.', ''): v for k, v in statedict.items()}
        # statedict = torch.load(f"/home/lrc/adp_gaze/Iter_6_gaze360.pt", map_location='cuda:1')

        net.to(device)
        net.load_state_dict(statedict)
        net.eval()
        # net.apply(apply_dropout)

        print(f"Test {saveiter}")
        length = len(dataset)
        accs = 0
        count = 0
        with torch.no_grad():
            with open(os.path.join(loadpath, f"evaluation_{folder[model_dir]}/{tests}/{saveiter}.log"), 'w') as outfile:
                outfile.write("name results gts\n")
                for j, (data, label) in enumerate(dataset):
                    img = data["face"].to(device)
                    names = data["name"]

                    img = {"face": img}
                    gts = label.to(device)

                    gazes, _ = net(img)
                    for k, gaze in enumerate(gazes):
                        gaze = gaze.cpu().detach().numpy()
                        count += 1
                        accs += angular(gaze, gts.cpu().numpy()[k])

                        name = [names[k]]
                        gaze = [str(u) for u in gaze]
                        gt = [str(u) for u in gts.cpu().numpy()[k]]
                        log = name + [",".join(gaze)] + [",".join(gt)]
                        outfile.write(" ".join(log) + "\n")

                loger = f"[{saveiter}] Total Num: {count}, avg: {accs / count}"
                outfile.write(loger)
                print(loger)
