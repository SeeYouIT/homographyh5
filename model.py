import torch.nn as nn
import torch.nn.functional as F


class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 16 * 16, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.contiguous().view(x.size(0), -1)
        out = self.fc(out)
        return out


class SportHomographyNet(nn.Module):
    def __init__(self):
        super(SportHomographyNet, self).__init__()
        self.corenet1 =  HomographyNet()
        self.corenet2 =  HomographyNet()
        self.corenet3 =  HomographyNet()
        self.corenet4 =  HomographyNet()
        self.corenet5 =  HomographyNet()


    def forward(self, x01,x12,x23,x34,x45,x10,x21,x32,x43,x54):
        x01 = self.corenet1(x01)
        x12 = self.corenet2(x12)
        x23 = self.corenet3(x23)
        x34 = self.corenet4(x34)
        x45 = self.corenet5(x45)
        x10 = self.corenet1(x10)
        x21 = self.corenet2(x21)
        x32 = self.corenet3(x32)
        x43 = self.corenet4(x43)
        x54 = self.corenet5(x54)

        return x01,x12,x23,x34,x45,x10,x21,x32,x43,x54



if __name__ == "__main__":
    '''
    from torchsummary import summary
    model = HomographyNet().cuda()
    print(HomographyNet())
    summary(model, (2, 128, 128))
    '''
    #from torchsummary import summary
    model = SportHomographyNet().cuda()
    #print(model)
    #summary(model, (2, 128, 128))
