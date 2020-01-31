import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
#transfer model
def transfer(block, pretrained):
    #load model
    base_model = models.vgg16(pretrained=pretrained)
    #redefine model 
    num2block = [4,9,16,23,30][block-1]
    base_vgg = base_model.features[:(num2block+1)]
    return base_vgg

#classificatioin model
def fcnet(n_size, fc_shapes, n_classes):
    layers = []
    in_shape = n_size
    for out_shape in fc_shapes:
        layers += [torch.nn.Linear(in_features=in_shape, out_features=out_shape),
                   torch.nn.ReLU(),
                   torch.nn.Dropout()
                   ]
        in_shape = out_shape
    classification = torch.nn.Linear(in_features=in_shape, out_features=n_classes)
    return torch.nn.ModuleList(layers), classification

class Network(torch.nn.Module):

    def __init__(self, block, input_shape, fc_shapes=[], n_classes=1, pretrained=True):
        super(Network, self).__init__()
        #vgg model
        self.base_vgg = transfer(block=block, pretrained=pretrained)
        #vgg model output shape
        self.n_size = self._get_conv_output(shape=input_shape)
        #classification model
        self.fc, self.disc = fcnet(n_size=self.n_size, fc_shapes=fc_shapes, n_classes=n_classes)
    
    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input_ = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input_)
        n_size = output_feat.data.size(1)
        return n_size

    def _forward_features(self, x):
        x = self.base_vgg(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, self.n_size)
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        x = self.disc(x)
        return x