import torch, utils
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners = False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners = False)
        return x

class conv_block(nn.Module):
    def __init__(self, channels_in, channels_out, kernel = 3, stride = 1, padding = 1):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel, stride, padding)
        self.inst = nn.InstanceNorm2d(channels_out)
        self.lrelu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        out = self.conv(x)
        out = self.inst(out)
        out = self.lrelu(out)
        return out
    
class dsconv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel = 3, stride = 1):
        super(dsconv, self).__init__()
        self.depthwise = nn.Conv2d(channels_in, channels_in, kernel, stride, 1, groups=channels_in)
        self.pointwise = nn.Conv2d(channels_in, channels_out, 1) #comes last in paper, after depthwise in code
        self.inst = nn.InstanceNorm2d(channels_out)
        self.lrelu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out) #move if necessary
        out = self.inst(out)
        out = self.lrelu(out)
        return out
    
class irb(nn.Module):
    def __init__(self, channels_in, channels_inter = None):
        super(irb, self).__init__()
        if channels_inter == None:
            channels_inter = channels_in * 2
        self.conv = conv_block(channels_in, channels_inter, 1, 1, 0)
        self.depthwise = nn.Conv2d(channels_inter, channels_inter, 3, 1, 1, groups=channels_inter)
        self.inst1 = nn.InstanceNorm2d(channels_in)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.pointwise = nn.Conv2d(channels_inter, channels_in, 1)
        self.inst2 = nn.InstanceNorm2d(channels_in)
    def forward(self, x):
        out = self.conv(x)
        out = self.depthwise(out)
        out = self.inst1(out)
        out = self.lrelu(out)
        out = self.pointwise(out)
        out = self.inst2(out)
        out = out + x
        return out

class downconv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(downconv, self).__init__()
        self.resize = Interpolate(scale_factor = 0.5, mode = 'bilinear', align_corners = False)
        self.halfdsconv = dsconv(channels_in, channels_out, stride = 2)
        self.samedsconv = dsconv(channels_in, channels_out)
    def forward(self, x):
        out1 = self.halfdsconv(x)
        out2 = self.resize(x)
        out2 = self.samedsconv(out2)
        out = out1 + out2
        return out

class upconv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(upconv, self).__init__()
        self.resize = Interpolate(scale_factor = 2, mode = 'bilinear', align_corners = False)
        self.dsconv = dsconv(channels_in, channels_out)
    def forward(self, x):
        out = self.resize(x)
        out = self.dsconv(out)
        return out
        
        
class generator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf = 64, nb = 8):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        
        self.down_convs = nn.Sequential(
            conv_block(3, nf),
            conv_block(nf, nf),
            downconv(nf, nf * 2),
            conv_block(nf * 2, nf * 2),
            dsconv(nf * 2, nf * 2),
            downconv(nf * 2, nf * 4),
            conv_block(nf * 4, nf * 4),
        )

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(irb(nf * 4))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.up_convs = nn.Sequential(
            conv_block(nf * 4, nf * 4),
            upconv(nf * 4, nf * 2),
            dsconv(nf * 2, nf * 2),
            conv_block(nf * 2, nf * 2),
            upconv(nf * 2, nf),
            conv_block(nf, nf),
            conv_block(nf, nf),
            nn.Conv2d(nf, 3, 1),
            nn.Tanh(),
        )

    # forward method
    def forward(self, input):
        x = self.down_convs(input)
        x = self.resnet_blocks(x)
        output = self.up_convs(x)
        return output


class discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(
            spectral_norm(nn.Conv2d(in_nc, nf, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf, nf * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 4, nf * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 8, out_nc, 3, 1, 1)),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        # input = torch.cat((input1, input2), 1)
        output = self.convs(input)

        return output


class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x
    
