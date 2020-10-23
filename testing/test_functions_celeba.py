
import os, sys

import numpy as np

import torch
from torch.autograd import Variable

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


import matplotlib.font_manager as fm
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from utils.constants_celeba import CLASSES_STR
from utils.constants_celeba import THRESHOLD_CLASSIFICATION
from utils import utils
from utils import plot
from utils.loss import calc_auc

from divergence_measures.mm_div import alpha_poe


def generate_random_samples_plot(flags, epoch, model, alphabet):
    img_size = torch.Size((3, flags.img_size, flags.img_size));
    random_samples = model.generate(100);
    img_samples = random_samples['img_celeba'];
    num_random_samples = random_samples['text'].shape[0];
    tensor_out_text = torch.zeros([int(num_random_samples), 3,
                                   flags.img_size, flags.img_size])
    text_samples = tensor_out_text.to(flags.device);
    for k in range(0, 100):
        text_samples[k,:,:,:] = plot.text_to_pil_celeba(random_samples['text'][k,:,:].unsqueeze(0), img_size, alphabet);
    fp_img = os.path.join(flags.dir_random_samples, 'random_epoch_' +
                          str(epoch).zfill(4) + 'img.png');
    fp_text = os.path.join(flags.dir_random_samples, 'random_epoch_' +
                           str(epoch).zfill(4) + 'text.png');
    plot_img = plot.create_fig(fp_img, img_samples, 10, flags.save_plot_images);
    plot_text = plot.create_fig(fp_text, text_samples, 10,
                                flags.save_plot_images);
    plots = {'img_celeba': plot_img, 'text': plot_text};
    return plots;


def generate_swapping_plot(flags, epoch, model, samples, alphabet):
    rec_i_in_i_out = Variable(torch.zeros([121, 3, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_i_in_t_out = Variable(torch.zeros([121, 3, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_i_out = Variable(torch.zeros([121, 3, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_t_out = Variable(torch.zeros([121, 3, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_i_in_i_out = rec_i_in_i_out.to(flags.device);
    rec_i_in_t_out = rec_i_in_t_out.to(flags.device);
    rec_t_in_i_out = rec_t_in_i_out.to(flags.device);
    rec_t_in_t_out = rec_t_in_t_out.to(flags.device);

    # ground truth: samples1 -> style (rows), samples2 -> content (cols)
    img_size = torch.Size((3, flags.img_size, flags.img_size));
    for i in range(len(samples)):
        c_text_sample = plot.text_to_pil_celeba(samples[i][1].unsqueeze(0), img_size, alphabet);
        c_img_sample = samples[i][0].squeeze();
        s_text_sample = c_text_sample.clone();
        s_img_sample = c_img_sample.clone();
        rec_i_in_i_out[i + 1, :, :, :] = c_img_sample
        rec_i_in_i_out[(i + 1) * 11, :, :, :] = s_img_sample;
        rec_i_in_t_out[i + 1, :, :, :] = c_img_sample
        rec_i_in_t_out[(i + 1) * 11, :, :, :] = s_text_sample;
        rec_t_in_i_out[i + 1, :, :, :] = c_text_sample
        rec_t_in_i_out[(i + 1) * 11, :, :, :] = s_img_sample;
        rec_t_in_t_out[i + 1, :, :, :] = c_text_sample
        rec_t_in_t_out[(i + 1) * 11, :, :, :] = s_text_sample;

    # style transfer
    for i in range(len(samples)):
        for j in range(len(samples)):
            l_style = model.inference(samples[i][0].unsqueeze(0),
                                      samples[i][1].unsqueeze(0));
            l_content = model.inference(samples[j][0].unsqueeze(0),
                                        samples[j][1].unsqueeze(0));

            l_c_img = l_content['img_celeba'];
            l_c_text = l_content['text'];
            l_s_img = l_style['img_celeba'];
            l_s_text = l_style['text'];
            s_img_emb = utils.reparameterize(mu=l_s_img[0], logvar=l_s_img[1]);
            c_img_emb = utils.reparameterize(mu=l_c_img[2], logvar=l_c_img[3]);
            s_text_emb = utils.reparameterize(mu=l_s_text[0], logvar=l_s_text[1]);
            c_text_emb = utils.reparameterize(mu=l_c_text[2], logvar=l_c_text[3]);
            style_emb = {'img_celeba': s_img_emb, 'text': s_text_emb};
            emb_c_img = {'content': c_img_emb, 'style': style_emb};
            emb_c_text = {'content': c_text_emb, 'style': style_emb};

            img_c_samples = model.generate_from_latents(emb_c_img);
            text_c_samples = model.generate_from_latents(emb_c_text);
            i_in_i_out = img_c_samples['img_celeba'];
            i_in_t_out = img_c_samples['text'];
            t_in_i_out = text_c_samples['img_celeba'];
            t_in_t_out = text_c_samples['text'];
            rec_i_in_i_out[(i+1)*11 + (j+1), :, :, :] = i_in_i_out;
            rec_i_in_t_out[(i+1)*11 + (j+1), :, :, :] = plot.text_to_pil_celeba(i_in_t_out, img_size, alphabet);
            rec_t_in_i_out[(i+1)*11 + (j+1), :, :, :] = t_in_i_out;
            rec_t_in_t_out[(i+1)*11 + (j+1), :, :, :] = plot.text_to_pil_celeba(t_in_t_out, img_size, alphabet);
    fp_i_in_i_out = os.path.join(flags.dir_swapping, 'swap_i_to_i_epoch_' + str(epoch).zfill(4) + '.png');
    fp_i_in_t_out = os.path.join(flags.dir_swapping, 'swap_i_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_i_out = os.path.join(flags.dir_swapping, 'swap_t_to_i_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_t_out = os.path.join(flags.dir_swapping, 'swap_t_to_t_epoch_' + str(epoch).zfill(4) + '.png');
    plot_i_i = plot.create_fig(fp_i_in_i_out, rec_i_in_i_out, 11,
                               flags.save_plot_images);
    plot_i_t = plot.create_fig(fp_i_in_t_out, rec_i_in_t_out, 11,
                               flags.save_plot_images);
    plot_t_i = plot.create_fig(fp_t_in_i_out, rec_t_in_i_out, 11,
                               flags.save_plot_images);
    plot_t_t = plot.create_fig(fp_t_in_t_out, rec_t_in_t_out, 11,
                               flags.save_plot_images);
    plots_c_img = {'img_celeba': plot_i_i, 'text': plot_i_t};
    plots_c_text = {'img_celeba': plot_t_i, 'text': plot_t_t};
    plots = {'img_celeba': plots_c_img, 'text': plots_c_text};
    return plots;


# modality defines the modality which is conditioned on (for the moment we restrict ourselves to condition on the content;
# as we are only looking at the bimodal case right now, it is clear, that the modality which is sampled from is the other one
# condition on content, sample from style, generate image from style modality

def generate_conditional_fig(flags, epoch, model, samples, alphabet):
    rec_i_in_i_out = Variable(torch.zeros([110, 3, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_i_in_t_out = Variable(torch.zeros([110, 3, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_i_out = Variable(torch.zeros([110, 3, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_t_in_t_out = Variable(torch.zeros([110, 3, flags.img_size, flags.img_size], dtype=torch.float32));
    rec_i_in_i_out = rec_i_in_i_out.to(flags.device);
    rec_i_in_t_out = rec_i_in_t_out.to(flags.device);
    rec_t_in_i_out = rec_t_in_i_out.to(flags.device);
    rec_t_in_t_out = rec_t_in_t_out.to(flags.device);
    # get style from random sampling
    zi_img = Variable(torch.randn(len(samples), flags.style_m1_dim)).to(flags.device);
    zi_text = Variable(torch.randn(len(samples), flags.style_m2_dim)).to(flags.device);

    # ground truth: samples1 -> style (rows), samples2 -> content (cols)
    img_size = torch.Size((3, flags.img_size, flags.img_size));
    for i in range(len(samples)):
        c_sample_text = plot.text_to_pil_celeba(samples[i][1].unsqueeze(0), img_size, alphabet);
        c_sample_img = samples[i][0].squeeze();
        rec_i_in_i_out[i, :, :, :] = c_sample_img;
        rec_i_in_t_out[i, :, :, :] = c_sample_img;
        rec_t_in_i_out[i, :, :, :] = c_sample_text;
        rec_t_in_t_out[i, :, :, :] = c_sample_text;

    # style transfer
    random_style = {'img_celeba': None, 'text': None};
    for i in range(len(samples)):
        for j in range(len(samples)):
            latents = model.inference(input_img=samples[j][0].unsqueeze(0),
                                      input_text=samples[j][1].unsqueeze(0))
            l_c_img = latents['img_celeba'][2:];
            l_c_text = latents['text'][2:];
            if flags.factorized_representation:
                random_style = {'img_celeba': zi_img[i].unsqueeze(0),
                                'text': zi_text[i].unsqueeze(0)};
            emb_c_img = utils.reparameterize(l_c_img[0], l_c_img[1]);
            emb_c_text = utils.reparameterize(l_c_text[0], l_c_text[1]);
            emb_img = {'content': emb_c_img, 'style': random_style};
            emb_text = {'content': emb_c_text, 'style': random_style};
            img_cond_gen = model.generate_from_latents(emb_img);
            text_cond_gen = model.generate_from_latents(emb_text);
            i_in_i_out = img_cond_gen['img_celeba'].squeeze(0);
            i_in_t_out = plot.text_to_pil_celeba(img_cond_gen['text'], img_size, alphabet);
            t_in_i_out = text_cond_gen['img_celeba'].squeeze(0);
            t_in_t_out = plot.text_to_pil_celeba(text_cond_gen['text'], img_size, alphabet);
            rec_i_in_i_out[(i+1)*10 + j, :, :, :] = i_in_i_out;
            rec_i_in_t_out[(i+1)*10 + j, :, :, :] = i_in_t_out;
            rec_t_in_i_out[(i+1)*10 + j, :, :, :] = t_in_i_out;
            rec_t_in_t_out[(i+1)*10 + j, :, :, :] = t_in_t_out;

    fp_i_in_i_out = os.path.join(flags.dir_cond_gen, 'cond_gen_img_img_epoch_' + str(epoch).zfill(4) + '.png');
    fp_i_in_t_out = os.path.join(flags.dir_cond_gen, 'cond_gen_img_text_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_i_out = os.path.join(flags.dir_cond_gen, 'cond_gen_text_img_epoch_' + str(epoch).zfill(4) + '.png');
    fp_t_in_t_out = os.path.join(flags.dir_cond_gen, 'cond_gen_text_text_epoch_' + str(epoch).zfill(4) + '.png');
    plot_i_i = plot.create_fig(fp_i_in_i_out, rec_i_in_i_out, 10,
                               flags.save_plot_images);
    plot_i_t = plot.create_fig(fp_i_in_t_out, rec_i_in_t_out, 10,
                               flags.save_plot_images);
    plot_t_i = plot.create_fig(fp_t_in_i_out, rec_t_in_i_out, 10,
                               flags.save_plot_images);
    plot_t_t = plot.create_fig(fp_t_in_t_out, rec_t_in_t_out, 10,
                               flags.save_plot_images);
    img_cond = {'img_celeba': plot_i_i, 'text': plot_i_t};
    text_cond = {'img_celeba': plot_t_i, 'text': plot_t_t};
    plots = {'img_celeba': img_cond, 'text': text_cond};
    return plots;


def classify_cond_gen_samples(flags, epoch, models, labels, samples):
    avg_precision = dict();
    auc = dict();
    for k, key in enumerate(models.keys()):
        mod_clf = models[key];
        mod_samples = samples[key]
        mod_attr = mod_clf(mod_samples);
        mod_pred_prob = mod_attr.cpu().data.numpy();
        mod_avg_prec_all_cl= dict();
        mod_auc_all_cl = dict();
        for l, label_str in enumerate(CLASSES_STR):
            mod_pred_prob_cl = mod_pred_prob[:,l];
            gt = labels[:,l].astype(int);
            mod_avg_precision = average_precision_score(gt,
            mod_pred_prob_cl.ravel());
            mod_fpr, mod_tpr, mod_auc = calc_auc(gt, mod_pred_prob_cl.ravel());
            mod_avg_prec_all_cl[label_str] = mod_avg_precision;
            mod_auc_all_cl[label_str] = mod_auc;
        avg_precision[key] = mod_avg_prec_all_cl;
        auc[key] = mod_auc_all_cl;
    return [auc, avg_precision];




def train_clfs_latent_representation(data, labels):
    clf_all = dict();
    for l, label_str in enumerate(CLASSES_STR):
        gt = labels[:, l].astype(int);
        clf_lr_label = dict()
        for k, key in enumerate(data.keys()):
            rep = data[key];
            clf_lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000);
            clf_lr.fit(rep, gt.ravel());
            clf_lr_label[key] = clf_lr;
        clf_all[label_str] = clf_lr_label;
    return clf_all;


def classify_latent_representations(flags, epoch, clf_lr, data, labels):
    ap_all_rep = dict();
    auc_all_rep = dict();
    for k, key in enumerate(data.keys()):
        ap_all_labels = dict();
        auc_all_labels = dict();
        for l, label_str in enumerate(CLASSES_STR):
            gt = labels[:,l].astype(int);
            clf_lr_label = clf_lr[label_str];
            clf_lr_rep = clf_lr_label[key];
            y_pred_rep = clf_lr_rep.predict_proba(data[key]);
            fpr_rep, tpr_rep, auc_rep = calc_auc(gt, y_pred_rep[:,1].ravel())
            ap_rep = average_precision_score(gt,y_pred_rep[:,1].ravel());
            ap_all_labels[label_str] = ap_rep;
            auc_all_labels[label_str] = auc_rep;
        auc_all_rep[key] = auc_all_labels;
        ap_all_rep[key] = ap_all_labels;
    return [auc_all_rep, ap_all_rep]



def classify_rand_gen_samples(flags, epoch, models, samples):
    model_clf_img = models['img_celeba'];
    model_clf_text = models['text'];
    random_img = samples['img_celeba'];
    random_text = samples['text'];
    attr_hat_img = model_clf_img(random_img);
    attr_hat_text = model_clf_text(random_text);
    pred_prob_gen_img = attr_hat_img.cpu().data.numpy();
    pred_prob_gen_text = attr_hat_text.cpu().data.numpy();

    coherence_all = dict()
    for k,label_str in enumerate(CLASSES_STR):
        pred_prob_gen_img_cl = pred_prob_gen_img[:,k];
        pred_prob_gen_text_cl = pred_prob_gen_text[:,k];
        pred_img = np.argmax(pred_prob_gen_img_cl).astype(int);
        pred_text = np.argmax(pred_prob_gen_text_cl).astype(int);
        coherence_cl = np.sum(pred_img == pred_text)/np.sum(np.ones(pred_img.shape));
        filename_coherence = os.path.join(flags.dir_gen_eval, label_str, 'random_coherence_epoch_' + str(epoch).zfill(4) + '.npy')
        np.save(filename_coherence, coherence_cl);
        coherence_all[label_str] = coherence_cl;
    return coherence_all;
