import torch
from torch.utils.data import Dataset
import os
import numpy as np

# Create a customized dataset class in pytorch
class CocoDdataset(Dataset):
    def __init__(self, path1, path2):
        lst = os.listdir(path1)
        self.data = [path2 + i for i in lst]

    def __getitem__(self, index):

        pic = self.data[index]
        head,end = pic.split('.')

        pic10 = head + '_10.npy'
        pic11 = head + '_11.npy'
        pic12 = head + '_12.npy'
        pic13 = head + '_13.npy'
        pic14 = head + '_14.npy'

        ori_images10 = np.load(pic10, allow_pickle=True)
        ori_images11 = np.load(pic11, allow_pickle=True)
        ori_images12 = np.load(pic12, allow_pickle=True)
        ori_images13 = np.load(pic13, allow_pickle=True)
        ori_images14 = np.load(pic14, allow_pickle=True)

        ori_images10 = np.transpose(ori_images10, [2, 0, 1])    # torch [C,H,W]
        ori_images11 = np.transpose(ori_images11, [2, 0, 1])    # torch [C,H,W]
        ori_images12 = np.transpose(ori_images12, [2, 0, 1])    # torch [C,H,W]
        ori_images13 = np.transpose(ori_images13, [2, 0, 1])    # torch [C,H,W]
        ori_images14 = np.transpose(ori_images14, [2, 0, 1])    # torch [C,H,W]

        ori_images0_1 = torch.from_numpy(ori_images10)
        ori_images1_2 = torch.from_numpy(ori_images11)
        ori_images2_3 = torch.from_numpy(ori_images12)
        ori_images3_4 = torch.from_numpy(ori_images13)
        ori_images4_5 = torch.from_numpy(ori_images14)


        pic20 = head + '_20.npy'
        pic21 = head + '_21.npy'
        pic22 = head + '_22.npy'
        pic23 = head + '_23.npy'
        pic24 = head + '_24.npy'

        ori_images20 = np.load(pic20, allow_pickle=True)
        ori_images21 = np.load(pic21, allow_pickle=True)
        ori_images22 = np.load(pic22, allow_pickle=True)
        ori_images23 = np.load(pic23, allow_pickle=True)
        ori_images24 = np.load(pic24, allow_pickle=True)

        ori_images20 = np.transpose(ori_images20, [2, 0, 1])    # torch [C,H,W]
        ori_images21 = np.transpose(ori_images21, [2, 0, 1])    # torch [C,H,W]
        ori_images22 = np.transpose(ori_images22, [2, 0, 1])    # torch [C,H,W]
        ori_images23 = np.transpose(ori_images23, [2, 0, 1])    # torch [C,H,W]
        ori_images24 = np.transpose(ori_images24, [2, 0, 1])    # torch [C,H,W]

        ori_images5_4 = torch.from_numpy(ori_images20)
        ori_images4_3 = torch.from_numpy(ori_images21)
        ori_images3_2 = torch.from_numpy(ori_images22)
        ori_images2_1 = torch.from_numpy(ori_images23)
        ori_images1_0 = torch.from_numpy(ori_images24)




        data_txt = head + '.txt'
        with open(data_txt, 'r') as f:

            line = f.readline()
            line = line.split(',')

            harm1 = torch.Tensor([[float(line[0]),float(line[1]),float(line[2])],[float(line[3]),float(line[4]),float(line[5])],[float(line[6]),float(line[7]),1.0]])
            harm2 = torch.Tensor([[float(line[8]),float(line[9]),float(line[10])],[float(line[11]),float(line[12]),float(line[13])],[float(line[14]),float(line[15]),1.0]])

            harm = [harm1,harm2]

        return ori_images0_1,ori_images1_2,ori_images2_3,ori_images3_4,ori_images4_5,ori_images1_0,ori_images2_1,ori_images3_2,ori_images4_3,ori_images5_4,harm

    def __len__(self):
        return len(self.data)
