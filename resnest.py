import torch.nn as nn

class Resnest(nn.Module):
    def __init__(self , model, num_classes):	# 此处的model参数是已经加载了预训练参数的模型
        super(Resnest, self).__init__()
        self.resnest_layer = nn.Sequential(*list(model.children())[:-2])  # 去掉model的后两层
        self.conv2d_1x1 = nn.Conv2d(2048, 1024, 1, 1)
        self.pool_layer = nn.AdaptiveAvgPool2d(1)
        self.Linear_layer = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        # features extractor (pretrained resnest)
        x = self.resnest_layer(x)   # -> bs,2048,8,8
        # do classification
        x = self.pool_layer(x)
        x = x.view(x.size(0), -1) 
        x = self.Linear_layer(x)
        return x

    def features(self, x):
        # features extractor (pretrained resnest)
        x = self.resnest_layer(x)   # -> bs,2048,8,8
        # 1x1 conv to adjust the channel number
        x = self.conv2d_1x1(x)      # -> bs,1024,8,8
        return x
