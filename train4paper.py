import torch
from torch import nn, optim
from datasetrain import CocoDdataset
from model import SportHomographyNet,HomographyNet
from torch.utils.data import DataLoader
import argparse
import time
import os
import cv2
import numpy as np
from mylossfunction import MyLossFunction

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def everyloss(outputs,a,b):
    a = torch.Tensor([a]).cuda()
    b = torch.Tensor([b]).cuda()
    O0=outputs[:,0]
    O1=outputs[:,1]
    O2=outputs[:,2]
    O3=outputs[:,3]
    O4=outputs[:,4]
    O5=outputs[:,5]
    O6=outputs[:,6]
    O7=outputs[:,7]

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

    return htmpout

def train(args):
    MODEL_SAVE_DIR = 'checkpoints/'
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    model = SportHomographyNet()
    model_dict = model.state_dict()

    model_path = os.path.join(MODEL_SAVE_DIR, args.checkpoint)
    modelchlid = HomographyNet()
    state = torch.load(model_path)
    modelchlid.load_state_dict(state['state_dict'])
    model_dictc = modelchlid.state_dict()

    for k in model_dictc.keys():
        kk1 = 'corenet1.'+k
        kk2 = 'corenet2.'+k
        kk3 = 'corenet2.'+k
        kk4 = 'corenet2.'+k
        kk5 = 'corenet2.'+k        
        model_dict[kk1] = model_dictc[k]
        model_dict[kk2] = model_dictc[k] 
        model_dict[kk3] = model_dictc[k]
        model_dict[kk4] = model_dictc[k] 
        model_dict[kk5] = model_dictc[k]    
    model.load_state_dict(model_dict)
    test = model.state_dict()

    TrainingData = CocoDdataset(args.train_path_index,args.train_path)
    ValidationData = CocoDdataset(args.val_path_index,args.val_path)
    print('Found totally {} training files and {} validation files'.format(len(TrainingData), len(ValidationData)))
    train_loader = DataLoader(TrainingData, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ValidationData, batch_size=args.batch_size, num_workers=0)


    if torch.cuda.is_available():
        model = model.cuda()
    mycriterion = MyLossFunction(1,10,0.01,10,1,0.1,10,10,1)
    criterion2 = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs / 2), gamma=0.1)

    print("start training")
    glob_iter = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Training
        model.train()
        train_loss = 0.0
        for i, batch_value in enumerate(train_loader):
            if (glob_iter % 150 == 0 and glob_iter != 0):
                filename = 'new_homographymodel' + '_iter_' + str(glob_iter) + '.pth'
                model_save_path = os.path.join(MODEL_SAVE_DIR, filename)
                state = {'epoch': args.epochs, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, model_save_path)
            
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

            target = batch_value[10]
            
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
        
            optimizer.zero_grad()
            outputs01,outputs12,outputs23,outputs34,outputs45,outputs10,outputs21,outputs32,outputs43,outputs54 = model(inputs01,inputs12,inputs23,inputs34,inputs45,inputs10,inputs21,inputs32,inputs43,inputs54)

            _h01 = everyloss(outputs01,128,128)
            _h12 = everyloss(outputs12,128,128)
            _h23 = everyloss(outputs23,128,128)
            _h34 = everyloss(outputs34,128,128)
            _h45 = everyloss(outputs45,128,128)

            _h10 = everyloss(outputs10,128,128)
            _h21 = everyloss(outputs21,128,128)
            _h32 = everyloss(outputs32,128,128)
            _h43 = everyloss(outputs43,128,128)
            _h54 = everyloss(outputs54,128,128)
            Hbe = target[0].cuda()
            Heb = target[1].cuda()

            h01 = _h01
            h12 = _h12
            h23 = _h23
            h34 = _h34
            h45 = _h45

            hr = torch.bmm(h45, h34)
            hr = torch.bmm(hr, h23)
            hr = torch.bmm(hr, h12) 
            hr = torch.bmm(hr, h01)

            lossr = mycriterion(hr,Hbe)

            h10 = _h10
            h21 = _h21
            h32 = _h32
            h43 = _h43
            h54 = _h54

            hl = torch.bmm(h10, h21)
            hl = torch.bmm(hl, h32)
            hl = torch.bmm(hl, h43)
            hl = torch.bmm(hl, h54)

            lossl = criterion(hl, Heb)

            e0 = torch.bmm(h01, h10)
            E0 = torch.eye(3).cuda()
            e1 = torch.bmm(h12, h21)
            E1 = torch.eye(3).cuda()
            e2 = torch.bmm(h23, h32)
            E2 = torch.eye(3).cuda()
            e3 = torch.bmm(h34, h43)
            E3 = torch.eye(3).cuda()
            e4 = torch.bmm(h45, h54)
            E4 = torch.eye(3).cuda()
            loss0 = criterion(E0, e0)
            loss1 = criterion(E1, e1)
            loss2 = criterion(E2, e2)
            loss3 = criterion(E3, e3)
            loss4 = criterion(E4, e4)

            lossm = loss1 + loss2 + loss3 + loss4 + loss0

            loss = lossr+lossm+lossl
            print("loss:",loss)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 20 == 0 or (i+1) == len(train_loader):
                print("Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Mean Squared Error: {:.4f} lr={:.6f}".format(
                    epoch+1, args.epochs, i+1, len(train_loader), train_loss / 2, scheduler.get_lr()[0]))
                train_loss = 0.0

            glob_iter += 1
        scheduler.step()


        if glob_iter%27==0:
            with torch.no_grad():
                model.eval()
                for k in model_dictc.keys():
                    #print(k)
                    kk1 = 'corenet1.'+k
                    kk2 = 'corenet2.'+k
                    kk3 = 'corenet1.'+k
                    kk4 = 'corenet2.'+k
                    kk5 = 'corenet1.'+k
                    sharepara = torch.true_divide(model_dict[kk1] + model_dict[kk2]+model_dict[kk3] + model_dict[kk4]+model_dict[kk5],5)
                    model_dict[kk1] = sharepara
                    model_dict[kk2] = sharepara  
                    model_dict[kk3] = sharepara
                    model_dict[kk4] = sharepara  
                    model_dict[kk5] = sharepara

    elapsed_time = time.time() - t0
    print("Finished Training in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))


if __name__ == "__main__":
    train_path = 'data/b/training/'
    val_path = 'data/b/validation/'
    train_path_index = 'data/b/training_index/'
    val_path_index = 'data/b/validation_index/'

    batch_size = 2
    num_samples = 500
    epochs = 2000

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=batch_size, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=epochs, help="number of epochs")
    parser.add_argument("--checkpoint", default="homographymodel.pth")
    parser.add_argument("--train_path", type=str, default=train_path, help="path to training imgs")
    parser.add_argument("--val_path", type=str, default=val_path, help="path to validation imgs")
    parser.add_argument("--train_path_index", type=str, default=train_path_index, help="path to training imgs")
    parser.add_argument("--val_path_index", type=str, default=val_path_index, help="path to validation imgs")
    args = parser.parse_args()
    train(args)