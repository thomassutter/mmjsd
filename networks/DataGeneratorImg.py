import torch.nn as nn

from networks.ResidualBlocks import ResidualBlock2dTransposeConv



def make_res_block_data_generator(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation, a_val=1.0, b_val=1.0):
    upsample = None;
    if (kernelsize != 1 and stride != 1) or (in_channels != out_channels):
        upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=o_padding),
                                 nn.BatchNorm2d(out_channels))
    layers = [];
    layers.append(ResidualBlock2dTransposeConv(in_channels, out_channels,
                                               kernelsize=kernelsize,
                                               stride=stride,
                                               padding=padding,
                                               dilation=dilation,
                                               o_padding=o_padding,
                                               upsample=upsample,
                                               a=a_val, b=b_val))
    return nn.Sequential(*layers)


def make_res_layers_data_generator(args, a=1.0, b=1.0):
    blocks = [];
    for k in range(0, args.num_layers_img):
        channels_in = min(args.num_layers_img, args.num_layers_img-k+1) * args.DIM_img;
        channels_out = (args.num_layers_img-k) * args.DIM_img;
        res_block = make_res_block_data_generator(channels_in, channels_out,
                                                  kernelsize=args.kernelsize_dec_img,
                                                  stride=args.dec_stride_img,
                                                  padding=args.dec_padding_img,
                                                  o_padding=args.dec_outputpadding_img,
                                                  dilation=1,
                                                  a_val=a,
                                                  b_val=b)
        blocks.append(res_block)
    return nn.Sequential(*blocks)


class DataGeneratorImg(nn.Module):
    def __init__(self, args, a, b):
        super(DataGeneratorImg, self).__init__()
        self.args = args;
        self.a = a;
        self.b = b;
        # self.data_generator = make_res_layers_data_generator(self.args, a=self.a, b=self.b)
        # self.resblock1 = make_res_block_data_generator(5*self.args.DIM_img, 5*self.args.DIM_img, kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.resblock1 = make_res_block_data_generator(5*self.args.DIM_img, 4*self.args.DIM_img, kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.resblock2 = make_res_block_data_generator(4*self.args.DIM_img, 3*self.args.DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.resblock3 = make_res_block_data_generator(3*self.args.DIM_img, 2*self.args.DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.resblock4 = make_res_block_data_generator(2*self.args.DIM_img, 1*self.args.DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.conv = nn.ConvTranspose2d(self.args.DIM_img, self.args.image_channels,
                                       kernel_size=self.args.kernelsize_dec_img,
                                       stride=self.args.dec_stride_img,
                                       padding=self.args.dec_padding_img,
                                       dilation=1,
                                       output_padding=self.args.dec_outputpadding_img);

    def forward(self, feats):
        # d = self.data_generator(feats)
        d = self.resblock1(feats);
        d = self.resblock2(d);
        d = self.resblock3(d);
        d = self.resblock4(d);
        # d = self.resblock5(d);
        d = self.conv(d)
        return d;