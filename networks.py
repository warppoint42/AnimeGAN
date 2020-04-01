import torch, utils
import torch.nn as nn
import torch.nn.functional as F


def add_depth_param(x, dp):
    if isinstance(x, int):
        x = (dp, x, x)
    elif len(x) == 2:
        x = (dp, x[0], x[1])
    return x   

class anim_resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        kernel = add_depth_param(kernel, 3)
        stride = add_depth_param(stride, 1)
        padding = add_depth_param(padding, 1)
        self.channel = channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Conv3d(channel, channel, kernel, stride, padding)
        self.conv1_norm = nn.InstanceNorm3d(channel)
        self.conv2 = nn.Conv3d(channel, channel, kernel, stride, padding)
        self.conv2_norm = nn.InstanceNorm3d(channel)

        utils.initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)), True)
        x = self.conv2_norm(self.conv2(x))

        return input + x #Elementwise Sum
    
def anim_conv(in_channels, out_channels, kernel_size, stride, padding):
    kernel_size = add_depth_param(kernel_size, 3)
    stride = add_depth_param(stride, 1)
    padding = add_depth_param(padding, 1)
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

def anim_convTranspose(in_channels, out_channels, kernel_size, stride, padding, output_padding):
    kernel_size = add_depth_param(kernel_size, 3)
    stride = add_depth_param(stride, 1)
    padding = add_depth_param(padding, 1)
    output_padding = add_depth_param(output_padding, 0)
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

class generator(nn.Module):
    
    # initializers
    def __init__(self, in_nc, out_nc, nf=32, nb=6):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        
        
        self.down_convs = nn.Sequential(
            anim_conv(in_nc, nf, 7, 1, 3), #k7n64s1
            nn.InstanceNorm3d(nf),
            nn.ReLU(True),
            anim_conv(nf, nf * 2, 3, 2, 1), #k3n128s2
            anim_conv(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm3d(nf * 2),
            nn.ReLU(True),
            anim_conv(nf * 2, nf * 4, 3, 2, 1), #k3n256s1
            anim_conv(nf * 4, nf * 4, 3, 1, 1), #k3n256s1
            nn.InstanceNorm3d(nf * 4),
            nn.ReLU(True),
        )

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(nf * 4, 3, 1, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.up_convs = nn.Sequential(
            anim_convTranspose(nf * 4, nf * 2, 3, 2, 1, 1), #k3n128s1/2
            anim_conv(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm3d(nf * 2),
            nn.ReLU(True),
            anim_convTranspose(nf * 2, nf, 3, 2, 1, 1), #k3n64s1/2
            anim_conv(nf, nf, 3, 1, 1), #k3n64s1
            nn.InstanceNorm3d(nf),
            nn.ReLU(True),
            anim_conv(nf, out_nc, 7, 1, 3), #k7n3s1
            nn.Tanh(),
        )

        utils.initialize_weights(self)

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
            anim_conv(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            anim_conv(nf, nf * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            anim_conv(nf * 2, nf * 4, 3, 1, 1),
            nn.InstanceNorm3d(nf * 4),
            nn.LeakyReLU(0.2, True),
            anim_conv(nf * 4, nf * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            anim_conv(nf * 4, nf * 8, 3, 1, 1),
            nn.InstanceNorm3d(nf * 8),
            nn.LeakyReLU(0.2, True),
            anim_conv(nf * 8, nf * 8, 3, 1, 1),
            nn.InstanceNorm3d(nf * 8),
            nn.LeakyReLU(0.2, True),
            anim_conv(nf * 8, out_nc, 3, 1, 1),
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
