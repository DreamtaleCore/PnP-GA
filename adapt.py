import model
import argparse
import reader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import torch.nn.functional as F
import random


def seed_everything(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def update_ema_params(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    seed_everything(1)
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--i', type=int, default=0,
                        help="i represents the i-th folder used as the test set")
    parser.add_argument('--significant', type=float, default=0.05, help="significance to judge outliers")
    parser.add_argument('--gamma', type=float, default=0.01, help="parameter in outlier loss")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--savepath', type=str, default=None, help="path to save models")
    parser.add_argument('--pics', type=int, default=-1, help="number of target images")
    parser.add_argument('--models', type=int, default=10, help="number of pretrained models(>1)")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--gpu', type=int, default=0, help="gpu")
    parser.add_argument('--source', type=str, default='eth', help="source dataset, eth/gaze360")
    parser.add_argument('--target', type=str, default='mpii', help="target dataset, mpii/edp")
    parser.add_argument('--shuffle', action='store_true', help="whether to shuffle")
    parser.add_argument('--alpha', type=float, default=0.1, help="weight of outlier loss")
    parser.add_argument('--beta', type=float, default=0.01, help="weight of KL")
    parser.add_argument('--sg', action='store_true', help="use source gaze constraint")
    parser.add_argument('--js', action='store_true', help="use js")
    parser.add_argument('--oma2', action='store_true', help="apply 2oma")
    args = parser.parse_args()

    config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    config = config["outlier"]

    loc = f'cuda:{args.gpu}'

    imagepath_target = config[args.target]["image"]
    labelpath_target = config[args.target]["label"]
    imagepath_source = config[args.source]["image"]
    labelpath_source = config[args.source]["label"]
    config['significant'] = args.significant
    config['alpha'] = args.alpha
    config['gamma'] = args.gamma
    alpha = args.alpha
    # modelname = config["save"]["model_name"]

    folder_target = os.listdir(labelpath_target)
    folder_target.sort()
    folder_source = os.listdir(labelpath_source)
    folder_source.sort()

    # i represents the i-th folder used as the test set.
    i = args.i
    pics_num = args.pics

    if i in list(range(len(folder_target))):
        tests_target = copy.deepcopy(folder_target)
        trains_target = [tests_target.pop(i)]
        print(f"Train Set:{trains_target}")
        trains_source = copy.deepcopy(folder_source)
    else:
        trains_target = copy.deepcopy(folder_target)
        print(f"Train Set:{trains_target}")
        trains_source = copy.deepcopy(folder_source)

    trainlabelpath_target = [os.path.join(labelpath_target, j) for j in trains_target]
    trainlabelpath_source = [os.path.join(labelpath_source, j) for j in trains_source]

    savepath = os.path.join(args.savepath, f"checkpoint/{trains_target[0]}")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    device = torch.device(loc if torch.cuda.is_available() else "cpu")

    print("Read data")
    dataset_target = reader.txtload(trainlabelpath_target, imagepath_target, args.batch_size,
                                    shuffle=args.shuffle, num_workers=4, pic_num=pics_num, header=True)
    dataset_source = reader.txtload(trainlabelpath_source, imagepath_source, args.batch_size,
                                    shuffle=args.shuffle, pic_num=pics_num, num_workers=4, header=True)

    print("Model building")
    pre_models = config[f'{args.source}_pretrains']
    if args.models > 1:
        pre_models = pre_models[:args.models]
    n = len(pre_models)
    print(f'Models num: {n}')
    net = [model.GazeRes18() for _ in range(n)]
    net_ema = [model.GazeRes18() for _ in range(n)]
    params = []
    for i in range(n):
        statedict = torch.load(pre_models[i], map_location=loc)
        net[i].to(device)
        net[i].load_state_dict(statedict)
        net[i].eval()
        net_ema[i].to(device)
        net_ema[i].load_state_dict(statedict)
        net_ema[i].eval()
        for value in net[i].parameters():
            if not value.requires_grad:
                continue
            params += [{'params': [value]}]
        for param in net_ema[i].parameters():
            param.detach_()

    print("optimizer building")
    outlier_loss_op = model.OutlierLoss().cuda()
    gaze_loss_op = nn.L1Loss().cuda()
    base_lr = args.lr

    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.Adam(params, lr=base_lr, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

    print("Traning")
    length_target = len(dataset_target)
    length_source = len(dataset_source)
    total = length_target * config["params"]["epoch"]
    cur = 0
    timebegin = time.time()
    with open(os.path.join(savepath, "train_log"), 'w') as outfile, \
            open(os.path.join(savepath, "loss_log"), 'w') as lossfile:
        for epoch in range(1, config["params"]["epoch"] + 1):
            for i, (target, source) in enumerate(zip(dataset_target, dataset_source)):
                # Acquire data
                data_target, _ = target
                data_source, label_source = source
                data_target["face"] = data_target["face"].to(device)
                data_source["face"] = data_source["face"].to(device)
                label_source = label_source.to(device).reshape(-1, 2, 1)

                # forward
                gaze_target = torch.Tensor().to(device)
                feature_target = torch.Tensor().to(device)
                for k in range(n):
                    gaze, feature = net[k](data_target)
                    gaze_target = torch.cat((gaze_target, gaze.reshape(-1, 2, 1)), 2)
                    feature_target = torch.cat((feature_target, feature.reshape(-1, 256, 1)), 2)

                gaze_source = torch.Tensor().to(device)
                for k in range(n):
                    gaze_source = torch.cat((gaze_source, net[k](data_source)[0].reshape(-1, 2, 1)), 2)

                gaze_ema_target = torch.Tensor().to(device)
                feature_ema_target = torch.Tensor().to(device)
                for k in range(n):
                    gaze_ema, feature_ema = net_ema[k](data_target)
                    gaze_ema_target = torch.cat((gaze_ema_target, gaze_ema.reshape(-1, 2, 1)), 2)
                    feature_ema_target = torch.cat((feature_ema_target, feature_ema.reshape(-1, 256, 1)), 2)

                gaze_ema_source = torch.Tensor().to(device)
                for k in range(n):
                    gaze_ema_source = torch.cat((gaze_ema_source, net_ema[k](data_source)[0].reshape(-1, 2, 1)), 2)

                # loss calculation
                outlier_loss = outlier_loss_op(gaze_target, gaze_ema_target, config['significant'], config['gamma'])
                loss = alpha * outlier_loss

                gaze_loss = 0
                if args.sg:
                    gaze_loss = gaze_loss_op(gaze_source, label_source)
                    loss += gaze_loss

                oma2_loss = 0
                if args.oma2:
                    oma2_loss = outlier_loss_op(gaze_ema_target, gaze_target, config['significant'], config['gamma'])
                    loss += alpha * oma2_loss

                js_loss = 0
                if args.js:
                    js_loss = F.kl_div(F.log_softmax(feature_target, dim=1),
                                       F.softmax(feature_ema_target, dim=1),
                                       reduction='mean') + F.kl_div(F.log_softmax(feature_ema_target, dim=1),
                                                                    F.softmax(feature_target, dim=1),
                                                                    reduction='mean')
                    loss += args.beta * 0.5 * js_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                cur += 1
                for k in range(n):
                    update_ema_params(net[k], net_ema[k], 0.99, epoch * total + i)

                # print logs
                if (i + 1) % 3 == 0 or True:
                    timeend = time.time()
                    resttime = (timeend - timebegin) / cur * (total - cur) / 3600
                    log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{min(length_source, length_target)}] " \
                          f"batch_size: {args.batch_size} js:{args.js} oma2:{args.oma2} sg:{args.sg} " \
                          f"pics:{pics_num} alpha:{alpha} " \
                          f"sig: {config['significant']} models: {args.models} gamma:{config['gamma']} " \
                          f"loss:{loss:.4f} gaze_loss:{gaze_loss:.4f} " \
                          f"oma2_loss:{oma2_loss:.4f} out_loss:{outlier_loss:.4f} js_loss:{js_loss:.4f} " \
                          f"lr:{optimizer.state_dict()['param_groups'][0]['lr']}, rest time:{resttime:.2f}h"
                    print(log)
                    outfile.write(log + "\n")
                    lossfile.write(f'{float(loss)} ')
                    sys.stdout.flush()
                    outfile.flush()
                    lossfile.flush()
            scheduler.step()
            lossfile.write(f'{float(loss)}\n')
            lossfile.flush()
            if epoch % config["save"]["step"] == 0:
                torch.save(net[2].state_dict(), os.path.join(savepath, f"Iter_{epoch}_GA.pt"))
