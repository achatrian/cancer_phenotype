import torch
import torch.nn as nn
from torchvision.models import inception_v3
import torch.utils.model_zoo as model_zoo
model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class Inception(torch.nn.Module):
    def __init__(self, num_classes):
        super(Inception, self).__init__()
        self.net = inception_v3(pretrained=False, num_classes=num_classes, aux_logits=True, transform_input=True)

        pretrained_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        model_dict = self.net.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['AuxLogits.fc.weight',
                                                                                 'AuxLogits.fc.bias',
                                                                                 'fc.weight',
                                                                                 'fc.bias']}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.net.load_state_dict(model_dict)

    def forward(self, x):
        x = self.net(x)
        return x