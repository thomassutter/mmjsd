import torch.nn as nn

from networks.ResidualBlocks import ResidualBlock1dTransposeConv


def make_res_block_decoder(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation, a_val=2.0, b_val=0.3):
    upsample = None;

    if (kernelsize != 1 or stride != 1) or (in_channels != out_channels) or dilation != 1:
        upsample = nn.Sequential(nn.ConvTranspose1d(in_channels, out_channels,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=o_padding),
                                   nn.BatchNorm1d(out_channels))
    layers = []
    layers.append(ResidualBlock1dTransposeConv(in_channels, out_channels, kernelsize, stride, padding, dilation, o_padding, upsample=upsample, a=a_val, b=b_val))
    return nn.Sequential(*layers)


def make_layers_resnet_decoder_data_generator(args, a=2.0, b=0.3):
    layers = [];
    for k in range(0,args.num_layers_text):
        channels_in = min(args.num_layers_text, args.num_layers_text-k+1) * args.DIM_text;
        channels_out = (args.num_layers_text-k) * args.DIM_text;
        resblock = make_res_block_decoder(channels_in, channels_out,
                                          kernelsize=args.kernelsize_dec_text,
                                          stride=args.dec_stride_text,
                                          padding=args.dec_padding_text,
                                          o_padding=args.dec_outputpadding_text,
                                          dilation=args.dec_dilation_text,
                                          a_val=a,
                                          b_val=b);
        layers.append(resblock)
    return nn.Sequential(*layers)

class DataGeneratorText(nn.Module):
    def __init__(self, args, a, b):
        super(DataGeneratorText, self).__init__()
        self.args = args
        self.a = a;
        self.b = b;
        self.resblock_1 = make_res_block_decoder(5*args.DIM_text, 5*args.DIM_text,
                                                 kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0);
        self.resblock_2 = make_res_block_decoder(5*args.DIM_text, 5*args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.resblock_3 = make_res_block_decoder(5*args.DIM_text, 4*args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.resblock_4 = make_res_block_decoder(4*args.DIM_text, 3*args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.resblock_5 = make_res_block_decoder(3*args.DIM_text, 2*args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.resblock_6 = make_res_block_decoder(2*args.DIM_text, args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.conv2 = nn.ConvTranspose1d(self.args.DIM_text, args.num_features,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        dilation=1,
                                        output_padding=0);
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, feats):
        d = self.resblock_1(feats);
        d = self.resblock_2(d);
        d = self.resblock_3(d);
        d = self.resblock_4(d);
        d = self.resblock_5(d);
        d = self.resblock_6(d);
        d = self.conv2(d)
        d = self.softmax(d);
        return d
