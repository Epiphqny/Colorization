import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)

class LowLevelFeatNet(nn.Module):
    def __init__(self):
        super(LowLevelFeatNet, self).__init__()
        self.features = nn.Sequential(
	        nn.Conv2d(1, 64, kernel_size = 3, stride = 2, padding = 1),
	        nn.BatchNorm2d(64),
	        nn.ReLU(),
	        nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
	        nn.BatchNorm2d(128),
	        nn.ReLU(),
	        nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1),
	        nn.BatchNorm2d(128),
	        nn.ReLU(),
	        nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
	        nn.BatchNorm2d(256),
	        nn.ReLU(),
	        nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1),
	        nn.BatchNorm2d(256),
	        nn.ReLU(),
	        nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
	        nn.BatchNorm2d(512),
	        nn.ReLU(),
        )
        self.apply(weights_init)


    def forward(self, x):
        x1=self.features(x)
        x2=x1.clone()
        return x1, x2


class MidLevelFeatNet(nn.Module):
    def __init__(self):
        super(MidLevelFeatNet, self).__init__()
        self.features = nn.Sequential(
	        nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
	        nn.BatchNorm2d(512),
	        nn.ReLU(),
	        nn.Conv2d(512, 256, kernel_size = 3, stride = 1, padding = 1),
	        nn.BatchNorm2d(256),
	        nn.ReLU(),
        )
        self.apply(weights_init)


    def forward(self, x):
        x=self.features(x)
        return x


class GlobalFeatNet(nn.Module):
    def __init__(self):
        super(GlobalFeatNet, self).__init__()
        self.features = nn.Sequential(
	        nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1),
	        nn.BatchNorm2d(512),
	        nn.ReLU(),
	        nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
	        nn.BatchNorm2d(512),
	        nn.ReLU(),
	        nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1),
	        nn.BatchNorm2d(512),
	        nn.ReLU(),
	        nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
	        nn.BatchNorm2d(512),
	        nn.ReLU(),
        )
        self.fc = nn.Sequential(
	        nn.Linear(25088, 1024),
	        nn.BatchNorm1d(1024),
	        nn.Linear(1024, 512),
	        nn.BatchNorm1d(512),
	        nn.Linear(512, 256),
	        nn.BatchNorm1d(256),
        )
        self.apply(weights_init)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 25088)
        output_256 = self.fc(x)
        return output_256



class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 313, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(313)
        # self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.apply(weights_init)

    def forward(self, mid_input, global_input):
        w = mid_input.size()[2]
        h = mid_input.size()[3]
        global_input = global_input.unsqueeze(2).unsqueeze(2).expand_as(mid_input)
        fusion_layer = torch.cat((mid_input, global_input), 1)
        fusion_layer = fusion_layer.permute(2, 3, 0, 1).contiguous()
        fusion_layer = fusion_layer.view(-1, 512)
        fusion_layer = self.bn1(self.fc1(fusion_layer))
        fusion_layer = fusion_layer.view(w, h, -1, 256)

        x = fusion_layer.permute(2, 3, 0, 1).contiguous()
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = self.upsample(x)
        x = F.relu(self.bn4(self.conv3(x)))
        # x=self.bn5(self.conv4(x))
        # x = F.sigmoid(self.bn5(self.conv4(x)))
        x = self.upsample(self.conv4(x))
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.low_lv_feat_net = LowLevelFeatNet()
        self.mid_lv_feat_net = MidLevelFeatNet()
        self.global_feat_net = GlobalFeatNet()
        # self.class_net = ClassificationNet()
        self.upsample_col_net = ColorizationNet()

    def forward(self, x):
        x1, x2 = self.low_lv_feat_net(x)
        # print('after low_lv, mid_input is:{}, global_input is:{}'.format(x1.size(), x2.size()))
        x1 = self.mid_lv_feat_net(x1)
        # print('after mid_lv, mid2fusion_input is:{}'.format(x1.size()))
        # class_input,\
        x2 = self.global_feat_net(x2)
        # print('after global_lv,  global2fusion_input is:{}'.format(x2.size()))
        # class_output = self.class_net(class_input)
        #print('after class_lv, class_output is:{}'.format(class_output.size()))
        output = self.upsample_col_net(x1, x2)
        # print('after upsample_lv, output is:{}'.format(output.size()))
        return output
