
import torch
import torch.nn as nn

from networks.FeatureExtractorText import FeatureExtractorText
from networks.FeatureCompressor import LinearFeatureCompressor
from networks.DataGeneratorText import DataGeneratorText


class EncoderText(nn.Module):
    def __init__(self, flags):
        super(EncoderText, self).__init__();
        self.feature_extractor = FeatureExtractorText(flags, a=flags.a_text, b=flags.b_text)
        self.feature_compressor = LinearFeatureCompressor(5*flags.DIM_text,
                                                          flags.style_m2_dim,
                                                          flags.class_dim)

    def forward(self, x_text):
        h_text = self.feature_extractor(x_text);
        mu_style, logvar_style, mu_content, logvar_content = self.feature_compressor(h_text);
        return mu_style, logvar_style, mu_content, logvar_content;


class DecoderText(nn.Module):
    def __init__(self, flags):
        super(DecoderText, self).__init__();
        self.feature_generator = nn.Linear(flags.style_m2_dim + flags.class_dim,
                                           5*flags.DIM_text, bias=True);
        self.text_generator = DataGeneratorText(flags, a=flags.a_text, b=flags.b_text)

    def forward(self, z_style, z_content):
        z = torch.cat((z_style, z_content), dim=1).squeeze(-1)
        text_feat_hat = self.feature_generator(z);
        text_feat_hat = text_feat_hat.unsqueeze(-1);
        text_hat = self.text_generator(text_feat_hat)
        text_hat = text_hat.transpose(-2,-1);
        return [text_hat];
