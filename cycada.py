import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torchvision

models = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def get_model(name, num_cls=10, **args):
    net = models[name](num_cls=num_cls, **args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net

"""
def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            print(m)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()
"""

def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()

class TaskNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    "Basic class which does classification."
    def __init__(self, num_cls=10, weights_init=None):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls
        self.setup_net()
        self.criterion = nn.CrossEntropyLoss()
        if weights_init is not None:
            self.load(weights_init)
        else:
            # import pdb;pdb.set_trace()
            init_weights(self)

    def forward(self, x, with_ft=False):
        x = self.conv_params(x) #NOTE input x shape: bs* 1* image_size*image_size
        x = x.view(x.size(0), -1)
        # import pdb; pdb.set_trace()
        x = self.fc_params(x)
        score = self.classifier(x) # bs * num_cls
        if with_ft:
            return score, x
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)
    
    def get_embedding_dim(self):
        return 512
    



@register_model('resnet18')
class ResNetClassifier(TaskNet):
    "Classifier used for VisDA2017 Experiment"

    num_channels = 3
    image_size = 32
    name = 'resnet18'
    out_dim = 512  # dim of last feature layer
    
    def get_embedding_dim(self):
        return out_dim
    
    def setup_net(self):
        model_resnet18 = torchvision.models.resnet18(pretrained=True)
        # pdb.set_trace()
        layers = [
                model_resnet18.conv1, 
                model_resnet18.bn1, 
                model_resnet18.relu,
                model_resnet18.maxpool,
                model_resnet18.layer1,
                model_resnet18.layer2,
                model_resnet18.layer3, # 1024
                model_resnet18.layer4, # 
                model_resnet18.avgpool
                ]

        self.conv_params = nn.Sequential(*layers)

        self.fc_params = nn.Sequential (
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                )

        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, self.num_cls)
                )
        self.get_embedding_dim = 512



@register_model('AddaNet')
class AddaNet(nn.Module):
    "Defines and Adda Network."
    def __init__(self, num_cls=10, model='LeNet', src_weights_init=None,
            weights_init=None):
        super(AddaNet, self).__init__()
        self.name = 'AddaNet'
        self.base_model = model
        self.num_cls = num_cls
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()
      
        self.setup_net()
        if weights_init is not None:
            self.load(weights_init)
        elif src_weights_init is not None:
            self.load_src_net(src_weights_init)
        else:
            raise Exception('AddaNet must be initialized with weights.')
        

    def forward(self, x_s, x_t):
        """Pass source and target images through their
        respective networks."""
        score_s, x_s = self.src_net(x_s, with_ft=True)
        score_t, x_t = self.tgt_net(x_t, with_ft=True)

        if self.discrim_feat:
            d_s = self.discriminator(x_s)
            d_t = self.discriminator(x_t)
        else:
            d_s = self.discriminator(score_s)
            d_t = self.discriminator(score_t)
        return score_s, score_t, d_s, d_t

    def setup_net(self):
        """Setup source, target and discriminator networks."""
        self.src_net = get_model(self.base_model, num_cls=self.num_cls)
        self.tgt_net = get_model(self.base_model, num_cls=self.num_cls)

        input_dim = self.num_cls 
        self.discriminator = nn.Sequential(
                nn.Linear(input_dim, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500, 2),
                )

        self.image_size = self.src_net.image_size
        self.num_channels = self.src_net.num_channels

    def load(self, init_path):
        "Loads full src and tgt models."
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def load_src_net(self, init_path):
        """Initialize source and target with source
        weights."""
        self.src_net.load(init_path)
        self.tgt_net.load(init_path)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def save_tgt_net(self, out_path):
        torch.save(self.tgt_net.state_dict(), out_path)