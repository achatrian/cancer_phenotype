import torch
from torchvision.models import inception_v3, resnet152, densenet169
import torch.utils.model_zoo as model_zoo
from pretrainedmodels import inceptionv4

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class Inception(torch.nn.Module):
    def __init__(self, num_classes):
        super(Inception, self).__init__()
        self.net = inceptionv4(pretrained=False, num_classes=num_classes)

        # if pretrained:
        #     pretrained_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        #     model_dict = self.net.state_dict()
        #
        #     # 1. filter out keys of layers to retrain
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['AuxLogits.fc.weight',
        #                                                                              'AuxLogits.fc.bias',
        #                                                                              'fc.weight',
        #                                                                              'fc.bias']}
        #     # 2. overwrite entries in the existing state dict
        #     model_dict.update(pretrained_dict)
        #     # 3. load the new state dict
        #     self.net.load_state_dict(model_dict)

    def forward(self, x):
        x = self.net(x)
        return x


class DenseNet(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(DenseNet, self).__init__()
        self.net = densenet169(pretrained=False, num_classes=num_classes)

        if pretrained:
            pretrained_dict = model_zoo.load_url(model_urls['densenet169'])
            model_dict = self.net.state_dict()

            # 1. filter out keys of layers to retrain
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['classifier.weight', 'classifier.bias']}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.net.load_state_dict(model_dict)

    def forward(self, x):
        x = self.net(x)
        return x


class ResNet(torch.nn.Module):
    def __init__(self, num_classes, pretrained):
        super(ResNet, self).__init__()
        self.net = resnet152(pretrained=False, num_classes=num_classes)

        if pretrained:
            pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
            model_dict = self.net.state_dict()

            # 1. filter out keys of layers to retrain
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['classifier.weight', 'classifier.bias']}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.net.load_state_dict(model_dict)

    def forward(self, x):
        x = self.net(x)
        return x