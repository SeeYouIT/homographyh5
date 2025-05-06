import torch
from torch import nn, optim
from datasetest import CocoDdataset
from model import SportHomographyNet,HomographyNet
from torch.utils.data import DataLoader
import argparse
import time
import os
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def everyloss(outputs,a,b):
    a = torch.Tensor([a]).cuda()
    b = torch.Tensor([b]).cuda()
    O0=outputs[:,0]
    O1=outputs[:,1]
    O2=outputs[:,2]-a
    O3=outputs[:,3]
    O4=outputs[:,4]
    O5=outputs[:,5]-b
    O6=outputs[:,6]-a
    O7=outputs[:,7]-b
    h7=(a*(O3-O7-b)*(O6+O0-O2-O4)-(O7+O1-O3-O5)*a*(O2-O6))/(a*(O3-O7-b)*b*(O4-O6-a)-b*(O5-O7)*a*(O2-O6))
    h6=(b*(O5-O7)*(O6+O0-O2-O4)-b*(O4-O6-a)*(O7+O1-O3-O5))/(b*(O5-O7)*a*(O2-O6)-b*(O4-O6-a)*a*(O3-O7-b))
    h5=O1
    h4=(O5+b-O1+b*(O5+b)*(a*(O3-O7-b)*(O6+O0-O2-O4)-(O7+O1-O3-O5)*a*(O2-O6))/(a*(O3-O7-b)*b*(O4-O6-a)-b*(O5-O7)*a*(O2-O6)))/b
    h3=(O3-O1+a*O3*(b*(O5-O7)*(O6+O0-O2-O4)-b*(O4-O6-a)*(O7+O1-O3-O5))/(b*(O5-O7)*a*(O2-O6)-b*(O4-O6-a)*a*(O3-O7-b)))/a
    h2=O0
    h1=(O4-O0+b*O4*(a*(O3-O7-b)*(O6+O0-O2-O4)-(O7+O1-O3-O5)*a*(O2-O6))/(a*(O3-O7-b)*b*(O4-O6-a)-b*(O5-O7)*a*(O2-O6)))/b
    h0=(O2+a-O0+a*(a+O2)*(b*(O5-O7)*(O6+O0-O2-O4)-b*(O4-O6-a)*(O7+O1-O3-O5))/(b*(O5-O7)*a*(O2-O6)-b*(O4-O6-a)*a*(O3-O7-b)))/a


    htmpout = torch.ones(args.batch_size,3,3).cuda()
    htmpout[:,0,0]=h0
    htmpout[:,0,1]=h1
    htmpout[:,0,2]=h2
    htmpout[:,1,0]=h3
    htmpout[:,1,1]=h4
    htmpout[:,1,2]=h5
    htmpout[:,2,0]=h6
    htmpout[:,2,1]=h7
    dertaHtmpout = torch.ones(args.batch_size,3,3).cuda()
    dertaHtmpout[:,0,0]=outputs[:,8]
    dertaHtmpout[:,0,1]=outputs[:,9]
    dertaHtmpout[:,0,2]=outputs[:,10]
    dertaHtmpout[:,1,0]=outputs[:,11]
    dertaHtmpout[:,1,1]=outputs[:,12]
    dertaHtmpout[:,1,2]=outputs[:,13]
    dertaHtmpout[:,2,0]=outputs[:,14]
    dertaHtmpout[:,2,1]=outputs[:,15]

    return htmpout,dertaHtmpout



def test(args):
    MODEL_SAVE_DIR = 'checkpoints/'
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_path = os.path.join(MODEL_SAVE_DIR, args.checkpoint)
    model = SportHomographyNet()
    state = torch.load(model_path)
    model.load_state_dict(state['state_dict'])

    TestData = CocoDdataset(args.test_path_index,args.test_path)
    print('Found totally {} test files'.format(len(TestData)))
    val_loader = DataLoader(TestData, batch_size=1, num_workers=0)

    if torch.cuda.is_available():
        model = model.cuda()

    print("start testing")
    t0 = time.time()
    for i, batch_value in enumerate(val_loader):

        inputs01 = batch_value[0].float()
        inputs12 = batch_value[1].float()
        inputs23 = batch_value[2].float()
        inputs34 = batch_value[3].float()
        inputs45 = batch_value[4].float()
        inputs10 = batch_value[5].float()
        inputs21 = batch_value[6].float()
        inputs32 = batch_value[7].float()
        inputs43 = batch_value[8].float()
        inputs54 = batch_value[9].float()

        if torch.cuda.is_available():
            inputs01 = inputs01.cuda()
            inputs12 = inputs12.cuda()
            inputs23 = inputs23.cuda()
            inputs34 = inputs34.cuda()
            inputs45 = inputs45.cuda()
            inputs10 = inputs10.cuda()
            inputs21 = inputs21.cuda()
            inputs32 = inputs32.cuda()
            inputs43 = inputs43.cuda()
            inputs54 = inputs54.cuda()

        outputs01,outputs12,outputs23,outputs34,outputs45,_,_,_,_,_ = model(inputs01,inputs12,inputs23,inputs34,inputs45,inputs10,inputs21,inputs32,inputs43,inputs54)

        #print(outputs01,outputs12,outputs23,outputs34,outputs45)
    elapsed_time = time.time() - t0
    print(elapsed_time)
    print("Finished Training in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))

if __name__ == "__main__":
    test_path = 'data/b/testing/'
    test_path_index = 'data/b/testing_index/'

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="new_homographymodel.pth")
    parser.add_argument("--test_path", type=str, default=test_path, help="path to testing imgs")
    parser.add_argument("--test_path_index", type=str, default=test_path_index, help="path to testing imgs")
    args = parser.parse_args()
    test(args)
