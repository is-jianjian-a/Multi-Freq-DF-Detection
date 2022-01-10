import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import pdb
from datetime import datetime
from torchviz import make_dot

from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
from sklearn.metrics import roc_auc_score


def ptime(index, steptime):
    print('epoch {} use {}'.format(index, datetime.now() - steptime))
    return datetime.now()


def main():
    args = parse.parse_args()
    method = args.method
    compressibility = args.compressibility
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epoches = args.epoches
    model_path = args.model_path
    num_workers = args.num_workers

    # 路径
    data_list = os.path.join("./data_list/", method + '/' + compressibility)
    tst_list = data_list + '/tst.txt'
    output_path = "./out_model/"
    pkl_path = os.path.join(output_path, method, compressibility)
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    # 打印本次任务所有信息
    print('Job begin at {}'.format(datetime.now()))
    print('Data = {}_{}'.format(method, compressibility))
    print('Epoch = {}'.format(epoches))
    print('Weight_decay = {}'.format(weight_decay))
    # 打印GPU信息
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ONLY using CPU")
    else:
        print("Let's use", torch.cuda.device_count(), "GPUs!")


    tst_dataset = MyDataset(txt_path=tst_list, transform=xception_default_data_transforms['train'])
    tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    tst_dataset_size = len(tst_dataset)

    # 模型和相关参数的选择与初始化
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    '''
    # 打印网络结构
    for (image, labels) in trn_loader:
        x = image
        x = x[0, :][np.newaxis, :]
        y = model(x)    # 获取网络的预测值​
        y = y[0, :]
        ModelVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x[0, :])]))
        ModelVis.format = "png"
        # 指定文件生成的文件夹
        ModelVis.directory = "data"
        # 生成文件
        ModelVis.view()
        break
    pdb.set_trace()
    '''
    model.load_state_dict(torch.load(model_path))
    model = nn.DataParallel(model)
    model.to(device)  # 用单个/多个GPU。直接代替model = model.cuda()

    # test_acc = 0.0
    iteration = 0
    label_list = []
    preds_list = []
    probs_list = []
    tst_corrects = 0.0

    # 测试部分
    model.eval()
    iteration = 0
    with torch.no_grad():
        for (image, labels) in tst_loader:
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            logits = nn.functional.softmax(outputs)  # 预测结果的概率形式
            probs = logits[:, 1]
            _, preds = torch.max(outputs.data, 1)
            label_list = label_list + labels.cpu().numpy().tolist()
            preds_list = preds_list + preds.cpu().numpy().tolist()
            probs_list = probs_list + probs.cpu().numpy().tolist()
            tst_corrects += torch.sum(preds == labels.data).to(torch.float32)
            iteration += 1
            if not (iteration % 400):
                print('Test iteration {:.4f} Acc: {:.4f}'.format(iteration, torch.sum(preds == labels.data).to(torch.float32) / batch_size))
        tst_acc = tst_corrects / tst_dataset_size
        preds_auc = roc_auc_score(label_list, preds_list)
        probs_auc = roc_auc_score(label_list, probs_list)
        print('Test Acc: {:.4f} Preds AUC: {:.4f} Prods AUC: {:.4f}'.format(tst_acc, preds_auc, probs_auc))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--method', '-m', type=str, default='NeuralTextures')  # 伪造方法，五选一
    parse.add_argument('--compressibility', '-c', type=str, default='c40')  # 压缩率，三选一
    parse.add_argument('--batch_size', '-bz', type=int, default=64)  # 1GPU选32，2GPU选64，4GPU选128
    parse.add_argument('--weight_decay', '-wd', type=float, default=1e-5)  # 权重衰减/L2正则化，防止过拟合
    parse.add_argument('--epoches', '-e', type=int, default='20')  # 20个epoches = 2天/GPU
    parse.add_argument('--model_path', '-mp', type=str, default='./out_model/NeuralTextures_c40_best.pkl')  # 继续训练的断点模型
    parse.add_argument('--num_workers', '-nw', type=int, default=8)  # 总线程数
    main()
