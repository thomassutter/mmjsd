import numpy as np

import torch.nn as nn

from networks.ResidualBlocks import ResidualBlock1dTransposeConv


def make_res_block_decoder_feature_generator(channels_in, channels_out, a_val=2.0, b_val=0.3):
    upsample = None;
    if channels_in != channels_out:
        upsample = nn.Sequential(nn.ConvTranspose1d(channels_in, channels_out,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    dilation=1,
                                                    output_padding=0),
                                   nn.BatchNorm1d(channels_out))
    layers = []
    layers.append(ResidualBlock1dTransposeConv(channels_in, channels_out,
                                               kernelsize=1,
                                               stride=1,
                                               padding=0,
                                               dilation=1,
                                               o_padding=0,
                                               upsample=upsample,
                                               a=a_val, b=b_val))
    return nn.Sequential(*layers)


def make_layers_resnet_decoder_feature_generator(start_channels, end_channels, a=2.0, b=0.3, l=1):
    layers = [];
    num_decompr_layers = int(1/float(l)*np.floor(np.log(end_channels / float(start_channels))))

    for k in range(0, num_decompr_layers):
        in_channels = start_channels*(2 ** (l*k))
        out_channels = start_channels*(2 ** (l*(k+1)))
        resblock = make_res_block_decoder_feature_generator(in_channels, out_channels, a_val=a, b_val=b);
        layers.append(resblock)
    if start_channels*(2 ** (l*num_decompr_layers)) < end_channels:
        resblock = make_res_block_decoder_feature_generator(start_channels*(2 ** (l*num_decompr_layers)), end_channels, a_val=a, b_val=b);
        layers.append(resblock)
    return nn.Sequential(*layers)

class FeatureGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, a, b, generation_power):
        super(FeatureGenerator, self).__init__()
        self.in_channels = in_channels;
        self.out_channels = out_channels;
        self.a = a;
        self.b = b;
        self.generation_power = generation_power;
        self.feature_generator = make_layers_resnet_decoder_feature_generator(self.in_channels,
                                                                              self.out_channels,
                                                                              a=self.a,
                                                                              b=self.b,
                                                                              l=self.generation_power)

    def forward(self, z):
        features = self.feature_generator(z);
        return features;