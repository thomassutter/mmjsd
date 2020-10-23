import sys
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from utils import utils
from utils import plot
from utils.constants_svhnmnist import indices
from utils.save_samples import write_samples_text_to_file
from divergence_measures.mm_div import alpha_poe, poe


transform_plot = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),
                                     transforms.ToTensor()])


def generate_random_samples_plots(flags, epoch, model, alphabet=None):
    img_size = torch.Size((3, flags.img_size_mnist, flags.img_size_mnist));
    random_samples = model.generate(100)
    samples_mnist = random_samples['img_mnist'];
    samples_svhn = random_samples['img_svhn'];
    fn_mnist = os.path.join(flags.dir_random_samples, 'random_epoch_' + str(epoch).zfill(4) + '_mnist.png');
    plot_mnist = plot.create_fig(fn_mnist, samples_mnist, 10,
                                 flags.save_plot_images);
    fn_svhn = os.path.join(flags.dir_random_samples, 'random_epoch_' + str(epoch).zfill(4) + '_svhn.png');
    plot_svhn = plot.create_fig(fn_svhn, samples_svhn, 10,
                                flags.save_plot_images);
    plots = {'img_mnist': plot_mnist, 'img_svhn': plot_svhn};
    if 'text' in random_samples.keys():
        num_random_samples = random_samples['text'].shape[0];
        tensor_out_text = torch.zeros([int(num_random_samples), 3,
                                       flags.img_size_mnist, flags.img_size_mnist])
        samples_text = tensor_out_text.to(flags.device);
        for k in range(0, num_random_samples):
            samples_text[k,:,::] = plot.text_to_pil(random_samples['text']
                                                    [k,:,:].unsqueeze(0), img_size, alphabet);
        fn_text = os.path.join(flags.dir_random_samples, 'random_epoch_' +
                               str(epoch).zfill(4) + '_text.png');
        plot_text = plot.create_fig(fn_text, samples_text, 10,
                                    flags.save_plot_images);
        plots['text'] = plot_text;
    return plots;



def generate_swapping_plot(flags, epoch, model, samples, alphabet=None):
    rec_m_in_m_out = Variable(torch.zeros([121, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_m_in_s_out = Variable(torch.zeros([121, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_m_in_t_out = Variable(torch.zeros([121, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_s_in_m_out = Variable(torch.zeros([121, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_s_in_s_out = Variable(torch.zeros([121, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_s_in_t_out = Variable(torch.zeros([121, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_t_in_m_out = Variable(torch.zeros([121, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_t_in_s_out = Variable(torch.zeros([121, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_t_in_t_out = Variable(torch.zeros([121, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_m_in_m_out = rec_m_in_m_out.to(flags.device);
    rec_m_in_s_out = rec_m_in_s_out.to(flags.device);
    rec_m_in_t_out = rec_m_in_t_out.to(flags.device);
    rec_s_in_m_out = rec_s_in_m_out.to(flags.device);
    rec_s_in_s_out = rec_s_in_s_out.to(flags.device);
    rec_s_in_t_out = rec_s_in_t_out.to(flags.device);
    rec_t_in_m_out = rec_t_in_m_out.to(flags.device);
    rec_t_in_s_out = rec_t_in_s_out.to(flags.device);
    rec_t_in_t_out = rec_t_in_t_out.to(flags.device);

    img_size = torch.Size((3, flags.img_size_mnist, flags.img_size_mnist));
    for i in range(len(samples)):
        c_sample_mnist = samples[i][0].squeeze().repeat(3, 1, 1);
        c_sample_svhn = samples[i][1].squeeze();
        s_sample_mnist = samples[i][0].squeeze().repeat(3, 1, 1);
        s_sample_svhn = samples[i][1].squeeze();
        if model.num_modalities == 3:
            c_sample_text = plot.text_to_pil(samples[i][2].unsqueeze(0), img_size, alphabet);
            s_sample_text = plot.text_to_pil(samples[i][2].unsqueeze(0), img_size, alphabet);
            rec_m_in_t_out[i+1, :, :, :] = c_sample_mnist;
            rec_m_in_t_out[(i + 1) * 11, :, :, :] = s_sample_text;
            rec_s_in_t_out[i+1, :, :, :] = transform_plot(c_sample_svhn.cpu()).cuda();
            rec_s_in_t_out[(i+1) * 11, :, :, :] = s_sample_text;
            rec_t_in_m_out[i+1, :, :, :] = c_sample_text;
            rec_t_in_m_out[(i + 1) * 11, :, :, :] = s_sample_mnist;
            rec_t_in_s_out[i+1, :, :, :] = c_sample_text;
            rec_t_in_s_out[(i + 1) * 11, :, :, :] = transform_plot(s_sample_svhn.cpu()).cuda();
            rec_t_in_t_out[i+1, :, :, :] = c_sample_text;
            rec_t_in_t_out[(i + 1) * 11, :, :, :] = s_sample_text;
        rec_m_in_m_out[i+1, :, :, :] = c_sample_mnist;
        rec_m_in_m_out[(i + 1) * 11, :, :, :] = s_sample_mnist;
        rec_m_in_s_out[i+1, :, :, :] = c_sample_mnist;
        rec_m_in_s_out[(i + 1) * 11, :, :, :] = transform_plot(s_sample_svhn.cpu()).cuda();
        rec_s_in_m_out[i+1, :, :, :] = transform_plot(c_sample_svhn.cpu()).cuda();
        rec_s_in_m_out[(i+1) * 11, :, :, :] = s_sample_mnist;
        rec_s_in_s_out[i+1, :, :, :] = transform_plot(c_sample_svhn.cpu()).cuda();
        rec_s_in_s_out[(i+1) * 11, :, :, :] = transform_plot(s_sample_svhn.cpu()).cuda();

    # style transfer
    for i in range(len(samples)):
        for j in range(len(samples)):
            if model.num_modalities == 2:
                latents_style = model.inference(samples[i][0].unsqueeze(0), samples[i][1].unsqueeze(0))
                latents_content = model.inference(samples[j][0].unsqueeze(0), samples[j][1].unsqueeze(0))
            elif model.num_modalities == 3:
                latents_style= model.inference(samples[i][0].unsqueeze(0),
                                                  samples[i][1].unsqueeze(0),
                                                  samples[i][2].unsqueeze(0))
                latents_content = model.inference(samples[j][0].unsqueeze(0),
                                                samples[j][1].unsqueeze(0),
                                                samples[j][2].unsqueeze(0))

            l_c_mnist = latents_content['img_mnist'];
            l_c_svhn = latents_content['img_svhn'];
            l_s_mnist = latents_style['img_mnist'];
            l_s_svhn = latents_style['img_svhn'];
            if model.num_modalities == 3:
                l_s_text = latents_style['text'];
                l_c_text = latents_content['text'];
                c_text_emb = utils.reparameterize(mu=l_c_text[2], logvar=l_c_text[3]);
                s_text_emb = utils.reparameterize(mu=l_s_text[0], logvar=l_s_text[1]);
            s_mnist_emb = utils.reparameterize(mu=l_s_mnist[0], logvar=l_s_mnist[1]);
            c_mnist_emb = utils.reparameterize(mu=l_c_mnist[2], logvar=l_c_mnist[3]);
            s_svhn_emb = utils.reparameterize(mu=l_s_svhn[0], logvar=l_s_svhn[1]);
            c_svhn_emb = utils.reparameterize(mu=l_c_svhn[2], logvar=l_c_svhn[3])
            if model.num_modalities == 3:
                style_emb = {'img_mnist': s_mnist_emb, 'img_svhn': s_svhn_emb, 'text': s_text_emb}
            else:
                style_emb = {'img_mnist': s_mnist_emb, 'img_svhn': s_svhn_emb}

            emb_content_mnist = {'content': c_mnist_emb, 'style': style_emb};
            emb_content_svhn = {'content': c_svhn_emb, 'style': style_emb}
            mnist_content_samples = model.generate_from_latents(emb_content_mnist);
            svhn_content_samples = model.generate_from_latents(emb_content_svhn);
            m_in_m_out = mnist_content_samples['img_mnist'];
            m_in_s_out = mnist_content_samples['img_svhn'];
            s_in_m_out = svhn_content_samples['img_mnist'];
            s_in_s_out = svhn_content_samples['img_svhn'];
            if model.num_modalities == 3:
                emb_content_text = {'content': c_text_emb, 'style': style_emb}
                text_content_samples = model.generate_from_latents(emb_content_text);
                m_in_t_out = mnist_content_samples['text'];
                s_in_t_out = svhn_content_samples['text'];
                t_in_m_out = text_content_samples['img_mnist'];
                t_in_s_out = text_content_samples['img_svhn'];
                t_in_t_out = text_content_samples['text'];

            rec_m_in_m_out[(i+1) * 11 + (j+1), :, :, :] = m_in_m_out.repeat(1, 3, 1, 1);
            rec_m_in_s_out[(i+1) * 11 + (j+1), :, :, :] = transform_plot(m_in_s_out.squeeze(0).cpu()).cuda().unsqueeze(0);
            rec_s_in_m_out[(i+1) * 11 + (j+1), :, :, :] = s_in_m_out.repeat(1, 3, 1, 1);
            rec_s_in_s_out[(i+1) * 11 + (j+1), :, :, :] = transform_plot(s_in_s_out.squeeze(0).cpu()).cuda().unsqueeze(0);
            if model.num_modalities == 3:
                rec_m_in_t_out[(i+1) * 11 + (j+1), :, :, :] = plot.text_to_pil(m_in_t_out, img_size, alphabet);
                rec_s_in_t_out[(i+1) * 11 + (j+1), :, :, :] = plot.text_to_pil(s_in_t_out, img_size, alphabet);
                rec_t_in_m_out[(i+1) * 11 + (j+1), :, :, :] = t_in_m_out.repeat(1, 3, 1, 1);
                rec_t_in_s_out[(i+1) * 11 + (j+1), :, :, :] = transform_plot(t_in_s_out.squeeze(0).cpu()).cuda().unsqueeze(0);
                rec_t_in_t_out[(i+1) * 11 + (j+1), :, :, :] = plot.text_to_pil(t_in_t_out, img_size, alphabet);

    fp_m_in_m_out = os.path.join(flags.dir_swapping, 'swap_m_to_m_epoch_' + str(epoch).zfill(4) + '.png');
    fp_m_in_s_out = os.path.join(flags.dir_swapping, 'swap_m_to_s_epoch_' + str(epoch).zfill(4) + '.png');
    fp_m_in_t_out = os.path.join(flags.dir_swapping, 'swap_m_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_s_in_m_out = os.path.join(flags.dir_swapping, 'swap_s_to_m_epoch_' + str(epoch).zfill(4) + '.png');
    fp_s_in_s_out = os.path.join(flags.dir_swapping, 'swap_s_to_s_epoch_' + str(epoch).zfill(4) + '.png');
    fp_s_in_t_out = os.path.join(flags.dir_swapping, 'swap_s_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_m_out = os.path.join(flags.dir_swapping, 'swap_s_to_m_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_s_out = os.path.join(flags.dir_swapping, 'swap_s_to_s_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_t_out = os.path.join(flags.dir_swapping, 'swap_s_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    plot_m_m = plot.create_fig(fp_m_in_m_out, rec_m_in_m_out, 11,
                               flags.save_plot_images);
    plot_m_s = plot.create_fig(fp_m_in_s_out, rec_m_in_s_out, 11,
                               flags.save_plot_images);
    plot_s_m = plot.create_fig(fp_s_in_m_out, rec_s_in_m_out, 11,
                               flags.save_plot_images);
    plot_s_s = plot.create_fig(fp_s_in_s_out, rec_s_in_s_out, 11,
                               flags.save_plot_images);
    plots_c_mnist = {'img_mnist': plot_m_m, 'img_svhn': plot_m_s};
    plots_c_svhn = {'img_mnist': plot_s_m, 'img_svhn': plot_s_s};
    plots = {'img_mnist': plots_c_mnist, 'img_svhn': plots_c_svhn};
    if model.num_modalities == 3:
        plot_m_t = plot.create_fig(fp_m_in_t_out, rec_m_in_t_out, 11,
                                   flags.save_plot_images);
        plots_c_mnist['text'] = plot_m_t;
        plot_s_t = plot.create_fig(fp_s_in_t_out, rec_s_in_t_out, 11,
                                   flags.save_plot_images);
        plots_c_svhn['text'] = plot_s_t;
        plot_t_m = plot.create_fig(fp_t_in_m_out, rec_t_in_m_out, 11,
                                   flags.save_plot_images);
        plot_t_s = plot.create_fig(fp_t_in_s_out, rec_t_in_s_out, 11,
                                   flags.save_plot_images);
        plot_t_t = plot.create_fig(fp_t_in_t_out, rec_t_in_t_out, 11,
                                   flags.save_plot_images);
        plots_c_text = {'img_mnist': plot_t_m, 'img_svhn': plot_t_s, 'text': plot_t_t};
        plots['text'] = plots_c_text;
    return plots;


def generate_conditional_fig_2a(flags, epoch, model, samples, alphabet=None):
    rec_mt_in_m_out = Variable(torch.zeros([120, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_mt_in_s_out = Variable(torch.zeros([120, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_mt_in_t_out = Variable(torch.zeros([120, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_ms_in_m_out = Variable(torch.zeros([120, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_ms_in_s_out = Variable(torch.zeros([120, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_ms_in_t_out = Variable(torch.zeros([120, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_st_in_m_out = Variable(torch.zeros([120, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_st_in_s_out = Variable(torch.zeros([120, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_st_in_t_out = Variable(torch.zeros([120, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_mt_in_m_out = rec_mt_in_m_out.to(flags.device);
    rec_mt_in_s_out = rec_mt_in_s_out.to(flags.device);
    rec_mt_in_t_out = rec_mt_in_t_out.to(flags.device);
    rec_ms_in_m_out = rec_ms_in_m_out.to(flags.device);
    rec_ms_in_s_out = rec_ms_in_s_out.to(flags.device);
    rec_ms_in_t_out = rec_ms_in_t_out.to(flags.device);
    rec_st_in_m_out = rec_st_in_m_out.to(flags.device);
    rec_st_in_s_out = rec_st_in_s_out.to(flags.device);
    rec_st_in_t_out = rec_st_in_t_out.to(flags.device);

    img_size = torch.Size((3, flags.img_size_mnist, flags.img_size_mnist));
    for i in range(len(samples)):
        c_sample_mnist = samples[i][0].squeeze().repeat(3, 1, 1);
        c_sample_svhn = transform_plot(samples[i][1].squeeze(0).cpu()).cuda().unsqueeze(0);
        c_sample_text = plot.text_to_pil(samples[i][2].unsqueeze(0), img_size, alphabet);
        rec_mt_in_m_out[i, :, :, :] = c_sample_mnist;
        rec_mt_in_m_out[i+10, :, :, :] = c_sample_text;
        rec_mt_in_s_out[i, :, :, :] = c_sample_mnist;
        rec_mt_in_s_out[i+10, :, :, :] = c_sample_text;
        rec_mt_in_t_out[i, :, :, :] = c_sample_mnist;
        rec_mt_in_t_out[i+10, :, :, :] = c_sample_text;
        rec_st_in_m_out[i, :, :, :] = c_sample_svhn;
        rec_st_in_m_out[i+10, :, :, :] = c_sample_text;
        rec_st_in_s_out[i, :, :, :] = c_sample_svhn;
        rec_st_in_s_out[i+10, :, :, :] = c_sample_text;
        rec_st_in_t_out[i, :, :, :] = c_sample_svhn;
        rec_st_in_t_out[i+10, :, :, :] = c_sample_text;
        rec_ms_in_m_out[i, :, :, :] = c_sample_mnist;
        rec_ms_in_m_out[i+10, :, :, :] = c_sample_svhn;
        rec_ms_in_s_out[i, :, :, :] = c_sample_mnist;
        rec_ms_in_s_out[i+10, :, :, :] = c_sample_svhn;
        rec_ms_in_t_out[i, :, :, :] = c_sample_mnist;
        rec_ms_in_t_out[i+10, :, :, :] = c_sample_svhn;

    # get style from random sampling
    zi_m = Variable(torch.randn(len(samples), flags.style_mnist_dim));
    zi_s = Variable(torch.randn(len(samples), flags.style_svhn_dim));
    zi_t = Variable(torch.randn(len(samples), flags.style_text_dim));
    zi_m = zi_m.to(flags.device);
    zi_s = zi_s.to(flags.device);
    zi_t = zi_t.to(flags.device);

    # style transfer
    for i in range(len(samples)):
        for j in range(len(samples)):
            l_ms = model.inference(input_mnist=samples[j][0].unsqueeze(0),
                                   input_svhn=samples[j][1].unsqueeze(0),
                                   input_text=None)
            l_mt = model.inference(input_mnist=samples[j][0].unsqueeze(0),
                                   input_svhn=None,
                                   input_text=samples[j][2].unsqueeze(0))
            l_st = model.inference(input_mnist=None,
                                   input_svhn=samples[j][1].unsqueeze(0),
                                   input_text=samples[j][2].unsqueeze(0))
            c_ms = l_ms['joint'];
            c_mt = l_mt['joint'];
            c_st = l_st['joint'];
            emb_ms_c = utils.reparameterize(c_ms[0], c_ms[1]);
            emb_mt_c = utils.reparameterize(c_mt[0], c_mt[1]);
            emb_st_c = utils.reparameterize(c_st[0], c_st[1]);

            if flags.factorized_representation:
                style = {'img_mnist': zi_m[i].unsqueeze(0),
                         'img_svhn': zi_s[i].unsqueeze(0),
                         'text': zi_t[i].unsqueeze(0)};
            else:
                style = {'img_mnist': None, 'img_svhn': None, 'text': None};
            emb_ms = {'content': emb_ms_c, 'style': style};
            emb_mt = {'content': emb_mt_c, 'style': style};
            emb_st = {'content': emb_st_c, 'style': style};
            ms_cond_gen = model.generate_from_latents(emb_ms);
            mt_cond_gen = model.generate_from_latents(emb_mt);
            st_cond_gen = model.generate_from_latents(emb_st);
            ms_in_m_out = ms_cond_gen['img_mnist'].repeat(1,3,1,1);
            ms_in_s_out = transform_plot(ms_cond_gen['img_svhn'].squeeze(0).cpu()).cuda().unsqueeze(0);
            ms_in_t_out = plot.text_to_pil(ms_cond_gen['text'], img_size, alphabet);
            mt_in_m_out = mt_cond_gen['img_mnist'].repeat(1,3,1,1);
            mt_in_s_out = transform_plot(mt_cond_gen['img_svhn'].squeeze(0).cpu()).cuda().unsqueeze(0);
            mt_in_t_out = plot.text_to_pil(mt_cond_gen['text'], img_size, alphabet);
            st_in_m_out = mt_cond_gen['img_mnist'].repeat(1,3,1,1);
            st_in_s_out = transform_plot(st_cond_gen['img_svhn'].squeeze(0).cpu()).cuda().unsqueeze(0);
            st_in_t_out = plot.text_to_pil(st_cond_gen['text'], img_size, alphabet);
            rec_ms_in_m_out[(i + 2) * 10 + j, :, :, :] = ms_in_m_out;
            rec_ms_in_s_out[(i + 2) * 10 + j, :, :, :] = ms_in_s_out;
            rec_ms_in_t_out[(i + 2) * 10 + j, :, :, :] = ms_in_t_out;
            rec_mt_in_m_out[(i + 2) * 10 + j, :, :, :] = mt_in_m_out;
            rec_mt_in_s_out[(i + 2) * 10 + j, :, :, :] = mt_in_s_out;
            rec_mt_in_t_out[(i + 2) * 10 + j, :, :, :] = mt_in_t_out;
            rec_st_in_m_out[(i + 2) * 10 + j, :, :, :] = st_in_m_out;
            rec_st_in_s_out[(i + 2) * 10 + j, :, :, :] = st_in_s_out;
            rec_st_in_t_out[(i + 2) * 10 + j, :, :, :] = st_in_t_out;

    fp_ms_m = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_ms_m_epoch_' + str(epoch).zfill(4) + '.png');
    fp_ms_s = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_ms_s_epoch_' + str(epoch).zfill(4) + '.png');
    fp_ms_t = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_ms_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_mt_m = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_mt_m_epoch_' + str(epoch).zfill(4) + '.png');
    fp_mt_s = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_mt_s_epoch_' + str(epoch).zfill(4) + '.png');
    fp_mt_t = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_mt_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_st_m = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_st_m_epoch_' + str(epoch).zfill(4) + '.png');
    fp_st_s = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_st_s_epoch_' + str(epoch).zfill(4) + '.png');
    fp_st_t = os.path.join(flags.dir_cond_gen_2a,
                           'cond_gen_st_t_epoch_' + str(epoch).zfill(4) + '.png');
    plot_ms_m = plot.create_fig(fp_ms_m, rec_ms_in_m_out, 10,
                                flags.save_plot_images);
    plot_ms_s = plot.create_fig(fp_ms_s, rec_ms_in_s_out, 10,
                                flags.save_plot_images);
    plot_ms_t = plot.create_fig(fp_ms_t, rec_ms_in_t_out, 10,
                                flags.save_plot_images);
    plot_mt_m = plot.create_fig(fp_mt_m, rec_mt_in_m_out, 10,
                                flags.save_plot_images);
    plot_mt_s = plot.create_fig(fp_mt_s, rec_mt_in_s_out, 10,
                                flags.save_plot_images);
    plot_mt_t = plot.create_fig(fp_mt_t, rec_mt_in_t_out, 10,
                                flags.save_plot_images);
    plot_st_m = plot.create_fig(fp_st_m, rec_st_in_m_out, 10,
                                flags.save_plot_images);
    plot_st_s = plot.create_fig(fp_st_s, rec_st_in_s_out, 10,
                                flags.save_plot_images);
    plot_st_t = plot.create_fig(fp_st_t, rec_st_in_t_out, 10,
                                flags.save_plot_images);
    plots_ms = {'img_mnist': plot_ms_m,
                'img_svhn': plot_ms_s,
                'text': plot_ms_t}
    plots_mt = {'img_mnist': plot_mt_m,
                'img_svhn': plot_mt_s,
                'text': plot_mt_t}
    plots_st = {'img_mnist': plot_st_m,
                'img_svhn': plot_st_s,
                'text': plot_st_t}
    plots = {'mnist_svhn': plots_ms,
             'mnist_text': plots_mt,
             'svhn_text': plots_st};
    return plots;


def generate_conditional_fig_1a(flags, epoch, model, samples, alphabet=None):
    rec_m_in_m_out = Variable(torch.zeros([110, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_m_in_s_out = Variable(torch.zeros([110, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_m_in_t_out = Variable(torch.zeros([110, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_s_in_m_out = Variable(torch.zeros([110, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_s_in_s_out = Variable(torch.zeros([110, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_s_in_t_out = Variable(torch.zeros([110, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_t_in_m_out = Variable(torch.zeros([110, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_t_in_s_out = Variable(torch.zeros([110, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    rec_t_in_t_out = Variable(torch.zeros([110, 3, flags.img_size_mnist, flags.img_size_mnist], dtype=torch.float32));
    # get style from random sampling
    rec_m_in_m_out = rec_m_in_m_out.to(flags.device);
    rec_m_in_s_out = rec_m_in_s_out.to(flags.device);
    rec_m_in_t_out = rec_m_in_t_out.to(flags.device);
    rec_s_in_m_out = rec_s_in_m_out.to(flags.device);
    rec_s_in_s_out = rec_s_in_s_out.to(flags.device);
    rec_s_in_t_out = rec_s_in_t_out.to(flags.device);
    rec_t_in_m_out = rec_t_in_m_out.to(flags.device);
    rec_t_in_s_out = rec_t_in_s_out.to(flags.device);
    rec_t_in_t_out = rec_t_in_t_out.to(flags.device);

    img_size = torch.Size((3, flags.img_size_mnist, flags.img_size_mnist));
    for i in range(len(samples)):
        c_sample_mnist = samples[i][0].squeeze().repeat(3, 1, 1);
        c_sample_svhn = samples[i][1].squeeze();
        if model.num_modalities == 3:
            c_sample_text = plot.text_to_pil(samples[i][2].unsqueeze(0), img_size, alphabet);
            rec_m_in_t_out[i, :, :, :] = c_sample_mnist;
            rec_s_in_t_out[i, :, :, :] = transform_plot(c_sample_svhn.squeeze(0).cpu()).cuda().unsqueeze(0);
            rec_t_in_m_out[i, :, :, :] = c_sample_text;
            rec_t_in_s_out[i, :, :, :] = c_sample_text;
            rec_t_in_t_out[i, :, :, :] = c_sample_text;
        rec_m_in_m_out[i, :, :, :] = c_sample_mnist;
        rec_m_in_s_out[i, :, :, :] = c_sample_mnist;
        rec_s_in_m_out[i, :, :, :] = transform_plot(c_sample_svhn.squeeze(0).cpu()).cuda().unsqueeze(0);
        rec_s_in_s_out[i, :, :, :] = transform_plot(c_sample_svhn.squeeze(0).cpu()).cuda().unsqueeze(0);

    # get style from random sampling
    zi_mnist = Variable(torch.randn(len(samples), flags.style_mnist_dim));
    zi_svhn = Variable(torch.randn(len(samples), flags.style_svhn_dim));
    zi_mnist = zi_mnist.to(flags.device);
    zi_svhn = zi_svhn.to(flags.device);
    if model.num_modalities == 3:
        zi_text = Variable(torch.randn(len(samples), flags.style_text_dim));
        zi_text = zi_text.to(flags.device);
    # style transfer
    for i in range(len(samples)):
        for j in range(len(samples)):
            # get content from given modalities
            if model.num_modalities == 2:
                latents = model.inference(samples[j][0].unsqueeze(0),
                                          samples[j][1].unsqueeze(0))
            else:
                latents = model.inference(samples[j][0].unsqueeze(0),
                                          samples[j][1].unsqueeze(0),
                                          samples[j][2].unsqueeze(0))

            c_mnist = latents['img_mnist'][2:];
            c_svhn = latents['img_svhn'][2:];
            mnist_rep = utils.reparameterize(mu=c_mnist[0], logvar=c_mnist[1]);
            svhn_rep = utils.reparameterize(mu=c_svhn[0], logvar=c_svhn[1]);
            if model.num_modalities == 3:
                c_text = latents['text'][2:];
                text_rep = utils.reparameterize(mu=c_text[0], logvar=c_text[1]);

            if flags.factorized_representation:
                style = {'img_mnist': zi_mnist[i].unsqueeze(0),
                         'img_svhn': zi_svhn[i].unsqueeze(0)};
                if model.num_modalities == 3:
                    style['text'] = zi_text[i].unsqueeze(0);
            else:
                style = {'img_mnist': None, 'img_svhn': None};
                if model.num_modalities:
                    style['text'] = None;
            cond_mnist = {'content': mnist_rep, 'style': style};
            cond_svhn = {'content': svhn_rep, 'style': style};
            m1_cond_gen_samples = model.generate_from_latents(cond_mnist);
            m2_cond_gen_samples = model.generate_from_latents(cond_svhn);

            m_in_m_out = m1_cond_gen_samples['img_mnist'];
            m_in_s_out = m1_cond_gen_samples['img_svhn'];
            s_in_m_out = m2_cond_gen_samples['img_mnist'];
            s_in_s_out = m2_cond_gen_samples['img_svhn'];
            if model.num_modalities == 3:
                cond_text = {'content': text_rep, 'style': style};
                m3_cond_gen_samples = model.generate_from_latents(cond_text);
                m_in_t_out = m1_cond_gen_samples['text'];
                s_in_t_out = m2_cond_gen_samples['text'];
                t_in_m_out = m3_cond_gen_samples['img_mnist'];
                t_in_s_out = m3_cond_gen_samples['img_svhn'];
                t_in_t_out = m3_cond_gen_samples['text'];
                fn_t = os.path.join(flags.dir_cond_gen_1a, 't_test_tensor_' + str(i) + '_' + str(j) + '.png');
                save_image(t_in_t_out.data.cpu(), fn_t, nrow=1);


            rec_m_in_m_out[(i+1) * 10 + j, :, :, :] = m_in_m_out.repeat(1,3,1,1);
            rec_m_in_s_out[(i+1) * 10 + j, :, :, :] = transform_plot(m_in_s_out.squeeze(0).cpu()).cuda().unsqueeze(0);
            rec_s_in_m_out[(i+1) * 10 + j, :, :, :] = s_in_m_out.repeat(1,3,1,1);
            rec_s_in_s_out[(i+1) * 10 + j, :, :, :] = transform_plot(s_in_s_out.squeeze(0).cpu()).cuda().unsqueeze(0);
            if model.num_modalities == 3:
                rec_m_in_t_out[(i+1) * 10 + j, :, :, :] = plot.text_to_pil(m_in_t_out, img_size, alphabet);
                rec_s_in_t_out[(i+1) * 10 + j, :, :, :] = plot.text_to_pil(s_in_t_out, img_size, alphabet);
                rec_t_in_m_out[(i+1) * 10 + j, :, :, :] = t_in_m_out.repeat(1,3,1,1);
                rec_t_in_s_out[(i+1) * 10 + j, :, :, :] = transform_plot(t_in_s_out.squeeze(0).cpu()).cuda().unsqueeze(0);
                rec_t_in_t_out[(i+1) * 10 + j, :, :, :] = plot.text_to_pil(t_in_t_out, img_size, alphabet);

    fp_m_in_m_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_m_to_m_epoch_' + str(epoch).zfill(4) + '.png');
    fp_m_in_s_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_m_to_s_epoch_' + str(epoch).zfill(4) + '.png');
    fp_m_in_t_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_m_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_s_in_m_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_s_to_m_epoch_' + str(epoch).zfill(4) + '.png');
    fp_s_in_s_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_s_to_s_epoch_' + str(epoch).zfill(4) + '.png');
    fp_s_in_t_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_s_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_m_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_t_to_m_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_s_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_t_to_s_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_t_out = os.path.join(flags.dir_cond_gen_1a, 'cond_gen_t_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    plot_m_m = plot.create_fig(fp_m_in_m_out, rec_m_in_m_out, 10,
                               flags.save_plot_images);
    plot_m_s = plot.create_fig(fp_m_in_s_out, rec_m_in_s_out, 10,
                               flags.save_plot_images);
    plot_s_m = plot.create_fig(fp_s_in_m_out, rec_s_in_m_out, 10,
                               flags.save_plot_images);
    plot_s_s = plot.create_fig(fp_s_in_s_out, rec_s_in_s_out, 10,
                               flags.save_plot_images);
    cond_mnist = {'img_mnist': plot_m_m, 'img_svhn': plot_m_s};
    cond_svhn = {'img_mnist': plot_s_m, 'img_svhn': plot_s_s};
    plots = {'img_mnist': cond_mnist, 'img_svhn': cond_svhn};
    if model.num_modalities == 3:
        plot_m_t = plot.create_fig(fp_m_in_t_out, rec_m_in_t_out, 10,
                                   flags.save_plot_images);
        cond_mnist['text'] = plot_m_t;
        plot_s_t = plot.create_fig(fp_s_in_t_out, rec_s_in_t_out, 10,
                                   flags.save_plot_images);
        cond_svhn['text'] = plot_s_t;
        plot_t_m = plot.create_fig(fp_t_in_m_out, rec_t_in_m_out, 10,
                                   flags.save_plot_images);
        plot_t_s = plot.create_fig(fp_t_in_s_out, rec_t_in_s_out, 10,
                                   flags.save_plot_images);
        plot_t_t = plot.create_fig(fp_t_in_t_out, rec_t_in_t_out, 10,
                                   flags.save_plot_images);
        cond_text = {'img_mnist': plot_t_m, 'img_svhn': plot_t_s, 'text': plot_t_t};
        plots['text'] = cond_text;
    return plots;


def classify_cond_gen_samples(flags, epoch, models, labels, cond_samples):
    gt = np.argmax(labels, axis=1).astype(int)
    mean_accuracy = dict();
    for key in models:
        if key in cond_samples:
            mod_cond_gen = cond_samples[key];
            mod_clf = models[key];
            attr_hat = mod_clf(mod_cond_gen);
            pred = np.argmax(attr_hat.cpu().data.numpy(), axis=1).astype(int);
            acc_mod = accuracy_score(gt, pred);
            mean_accuracy[key] = acc_mod;
        else:
            print(str(key) + 'not existing in cond_gen_samples');
    return mean_accuracy


def classify_latent_representations(flags, epoch, clf_lr, data, labels):
    gt = np.argmax(labels, axis=1).astype(int)
    accuracies = dict()
    for key in clf_lr:
        data_rep = data[key];
        clf_lr_rep = clf_lr[key];
        y_pred_rep = clf_lr_rep.predict(data_rep);
        accuracy_rep = accuracy_score(gt, y_pred_rep.ravel());
        cm_rep = confusion_matrix(gt, y_pred_rep);
        accuracies[key] = accuracy_rep;
    return accuracies;


def train_clf_lr(flags, data, labels):
    gt = np.argmax(labels, axis=1).astype(int)
    clf_lr = dict();
    for k, key in enumerate(data.keys()):
        data_rep = data[key];
        clf_lr_rep = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000);
        clf_lr_rep.fit(data_rep, gt.ravel());
        clf_lr[key] = clf_lr_rep;
    return clf_lr;


def calculate_coherence(models, samples):
    model_clf_mnist = models['img_mnist'];
    model_clf_svhn = models['img_svhn'];
    mnist_rand_gen = samples['img_mnist'];
    svhn_rand_gen = samples['img_svhn'];
    attr_hat_mnist = model_clf_mnist(mnist_rand_gen);
    attr_hat_svhn = model_clf_svhn(svhn_rand_gen);
    output_prob_mnist = attr_hat_mnist.cpu().data.numpy();
    output_prob_svhn = attr_hat_svhn.cpu().data.numpy();
    pred_mnist = np.argmax(output_prob_mnist, axis=1).astype(int);
    pred_svhn = np.argmax(output_prob_svhn, axis=1).astype(int);
    if 'text' in models.keys():
        model_clf_text = models['text'];
        text_rand_gen = samples['text'];
        attr_hat_text = model_clf_text(text_rand_gen);
        output_prob_text = attr_hat_text.cpu().data.numpy();
        pred_text = np.argmax(output_prob_text, axis=1).astype(int);
        coherence_m1_m2 = (pred_mnist == pred_svhn);
        coherence_m1_m3 = (pred_mnist == pred_text);
        coherence = np.sum(coherence_m1_m2 == coherence_m1_m3) / np.sum(np.ones(pred_mnist.shape));
    else:
        coherence = np.sum(pred_mnist == pred_svhn) / np.sum(np.ones(pred_mnist.shape));
    return coherence;

