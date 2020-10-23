import os
import numpy as np
from itertools import cycle
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from divergence_measures.kl_div import calc_kl_divergence
from divergence_measures.mm_div import alpha_poe

from testing.test_functions_celeba import generate_swapping_plot
from testing.test_functions_celeba import generate_conditional_fig
from testing.test_functions_celeba import generate_random_samples_plot

from testing.test_functions_celeba import classify_cond_gen_samples
from testing.test_functions_celeba import classify_latent_representations
from testing.test_functions_celeba import classify_rand_gen_samples
from testing.test_functions_celeba import train_clfs_latent_representation
from utils.test_functions import load_inception_activations
from utils.test_functions import calculate_inception_features_for_gen_evaluation
from utils.test_functions import calculate_fid
from utils.test_functions import calculate_prd
from fid.fid_score import calculate_fid_given_paths

from utils import utils
from utils import save_samples
from utils.loss import log_prob_img, log_prob_text
from utils.save_samples import save_generated_samples_singlegroup
from utils.constants_celeba import CLASSES_STR



torch.multiprocessing.set_sharing_strategy('file_system')

# global variables
NUM_ATTRIBUTES = 42;


def create_dict_all_labels():
    all_labels = dict();
    for k, key in enumerate(CLASSES_STR):
       all_labels[key] = [];
    return all_labels;


def add_mean_all_labels(dict_list_all, values_labels):
    for k, key in enumerate(dict_list_all.keys()):
        l_key = dict_list_all[key];
        v_key = values_labels[key];
        l_key.append(np.mean(v_key))
        dict_list_all[key] = l_key;
    return dict_list_all;


def get_mean_all_labels(dict_list_all):
   for k, key in enumerate(dict_list_all.keys()):
       l_key = dict_list_all[key];
       dict_list_all[key] = np.mean(np.array(l_key));
   return dict_list_all;


def run_epoch(epoch, vae_bimodal, optimizer, data, writer, alphabet,
              test_samples=None, train=False, flags={},
              model_clf_img=None, model_clf_text = None,
              clf_lr=None, step_logs=0):

    loader = cycle(DataLoader(data, batch_size=flags.batch_size, shuffle=True, num_workers=8, drop_last=True))

    #set up weights
    beta_style = flags.beta_style;
    beta_content = flags.beta_content;

    beta_m1_style = flags.beta_m1_style;
    beta_m2_style = flags.beta_m2_style;

    rec_weight_m1 = vae_bimodal.rec_w1;
    rec_weight_m2 = vae_bimodal.rec_w2;

    beta = flags.beta;
    rec_weight = 1.0;

    if not train:
        vae_bimodal.eval();
        lr_ap_m1 = create_dict_all_labels();
        lr_ap_m2 = create_dict_all_labels();
        lr_ap_joint = create_dict_all_labels();
        lr_ap_m1_s = create_dict_all_labels();
        lr_ap_m2_s = create_dict_all_labels();
        cg_ap_m1 = {'img_celeba': create_dict_all_labels(),
                    'text': create_dict_all_labels()};
        cg_ap_m2 = {'img_celeba': create_dict_all_labels(),
                    'text': create_dict_all_labels()};
        random_coherence = create_dict_all_labels();
    else:
        vae_bimodal.train();

    num_batches_epoch = int(data.img_names.shape[0] / flags.batch_size)
    step_print_progress = 0;
    for iteration in range(num_batches_epoch):
        # load a mini-batch
        m1_batch, m2_batch, labels_batch = next(loader)
        m1_batch = Variable(m1_batch).to(flags.device);
        m2_batch = Variable(m2_batch).to(flags.device);
        labels_batch = Variable(labels_batch).to(flags.device);

        results_joint = vae_bimodal(Variable(m1_batch), Variable(m2_batch));
        m1_reconstruction = results_joint['rec']['img_celeba'];
        m2_reconstruction = results_joint['rec']['text'];
        [m1_style_mu, m1_style_logvar, m1_class_mu, m1_class_logvar] = results_joint['latents']['img_celeba'];
        [m2_style_mu, m2_style_logvar, m2_class_mu, m2_class_logvar] = results_joint['latents']['text'];
        [group_mu, group_logvar] = results_joint['group_distr'];
        group_divergence = results_joint['joint_divergence'];
        if flags.modality_jsd:
            [dyn_prior_mu, dyn_prior_logvar] = results_joint['dyn_prior'];
            kld_dyn_prior = calc_kl_divergence(dyn_prior_mu, dyn_prior_logvar, norm_value=flags.batch_size)

        if flags.factorized_representation:
            kld_m1_style = calc_kl_divergence(m1_style_mu, m1_style_logvar, norm_value=flags.batch_size)
            kld_m2_style = calc_kl_divergence(m2_style_mu, m2_style_logvar, norm_value=flags.batch_size)
        else:
            m1_style_mu = torch.zeros(1);
            m1_style_logvar = torch.zeros(1);
            m2_style_mu = torch.zeros(1);
            m2_style_logvar = torch.zeros(1);
            kld_m1_style = torch.zeros(1);
            kld_m2_style = torch.zeros(1);
            m1_style_mu = m1_style_mu.to(flags.device);
            m1_style_logvar = m1_style_logvar.to(flags.device);
            m2_style_mu = m2_style_mu.to(flags.device);
            m2_style_logvar = m2_style_logvar.to(flags.device);
            kld_m1_style = kld_m1_style.to(flags.device);
            kld_m2_style = kld_m2_style.to(flags.device);

        kld_m1_class = calc_kl_divergence(m1_class_mu, m1_class_logvar, norm_value=flags.batch_size);
        kld_m2_class = calc_kl_divergence(m2_class_mu, m2_class_logvar, norm_value=flags.batch_size);
        kld_group = calc_kl_divergence(group_mu, group_logvar, norm_value=flags.batch_size);
        rec_error_m1 = -log_prob_img(m1_reconstruction, Variable(m1_batch), norm_value=flags.batch_size);
        rec_error_m2 = -log_prob_text(m2_reconstruction, Variable(m2_batch), norm_value=flags.batch_size);

        rec_error_weighted = rec_weight_m1 * rec_error_m1 + rec_weight_m2 * rec_error_m2;
        if flags.modality_moe or flags.modality_jsd:
            kld_style = beta_m1_style * kld_m1_style + beta_m2_style * kld_m2_style;
            kld_content = group_divergence;
            kld_weighted_all = beta_style * kld_style + beta_content * kld_content;
            total_loss = rec_weight*rec_error_weighted + beta*kld_weighted_all
        elif flags.modality_poe:
            klds_joint = {'content': group_divergence,
                          'style': {'img_celeba': kld_m1_style, 'text': kld_m2_style}}
            recs_joint = {'img_celeba': rec_error_m1, 'text': rec_error_m2};
            elbo_joint = utils.calc_elbo_celeba(flags, 'joint', recs_joint,
                                                klds_joint);
            results_img = vae_bimodal(input_img=Variable(m1_batch),
                                      input_text=None);
            img_m1_rec = results_img['rec']['img_celeba'];
            klds_img = {'content': kld_m1_class,
                        'style':{'img_celeba': kld_m1_style}}
            img_m1_rec_error = -log_prob_img(img_m1_rec,
                                             Variable(m1_batch),
                                             flags.batch_size);
            recs_img = {'img_celeba': img_m1_rec_error};
            elbo_img = utils.calc_elbo_celeba(flags, 'img_celeba', recs_img,
                                       klds_img);

            results_text = vae_bimodal(input_img=None,
                                       input_text=Variable(m2_batch));
            text_m2_rec = results_text['rec']['text'];
            klds_text = {'content': kld_m2_class,
                         'style': {'text': kld_m2_style}};
            text_m2_rec_error = -log_prob_text(text_m2_rec,
                                               Variable(m2_batch),
                                               flags.batch_size)
            recs_text = {'text': text_m2_rec_error};
            elbo_text = utils.calc_elbo_celeba(flags, 'text', recs_text,
                                               klds_text);
            total_loss = elbo_joint + elbo_img + elbo_text;

        if flags.unimodal_klds:
            kld_content = (1 / 3) * group_divergence + (1 / 3) * kld_m1_class + (1 / 3) * kld_m2_class;
        else:
            kld_content = group_divergence;


        data_class_m1 = m1_class_mu.cpu().data.numpy();
        data_class_m2 = m2_class_mu.cpu().data.numpy();
        data_class_joint = group_mu.cpu().data.numpy();
        data = {'img': data_class_m1,
                'text': data_class_m2,
                'joint': data_class_joint,
                }
        if flags.factorized_representation:
            data_style_m1 = m1_style_mu.cpu().data.numpy();
            data_style_m2 = m2_style_mu.cpu().data.numpy();
            data['style_img'] = data_style_m1;
            data['style_text'] = data_style_m2;
        labels = labels_batch.cpu().data.numpy();
        if (epoch + 1) % flags.eval_freq == 0 or (epoch + 1) == flags.end_epoch:
            if train == False:
                # conditional generation
                latent_distr = dict();
                latent_distr['img_celeba'] = [m1_class_mu, m1_class_logvar];
                latent_distr['text'] = [m2_class_mu, m2_class_logvar];
                rand_gen_samples = vae_bimodal.generate(flags.batch_size);
                cond_gen_samples = vae_bimodal.cond_generation(latent_distr);
                m1_cond = cond_gen_samples['img_celeba']  # samples conditioned on img;
                m2_cond = cond_gen_samples['text']  # samples conditioned on text;
                m1_cond_gen = m2_cond['img_celeba'];
                m2_cond_gen = m1_cond['text'];
                real_samples = {'img_celeba': m1_batch, 'text': m2_batch};
                save_generated_samples_singlegroup(flags, iteration, alphabet, 'real', real_samples)
                save_generated_samples_singlegroup(flags, iteration, alphabet, 'random_sampling', rand_gen_samples)
                cond_samples = {'img_celeba': m1_cond_gen, 'text': m2_cond_gen};
                save_generated_samples_singlegroup(flags, iteration, alphabet, 'cond_gen', cond_samples)

                if flags.use_clf and model_clf_img is not None and model_clf_text is not None:
                    model_dict = {'img_celeba': model_clf_img, 'text': model_clf_text};
                    cond_gen_samples = {'img_celeba': m1_cond_gen, 'text': m2_cond_gen};
                    ap_cond_img = classify_cond_gen_samples(flags, epoch,
                                                            model_dict,
                                                            labels,
                                                            m1_cond)[-1];
                    ap_cond_text = classify_cond_gen_samples(flags, epoch,
                                                             model_dict,
                                                             labels,
                                                             m2_cond)[-1];
                    cg_ap_m1['img_celeba'] = add_mean_all_labels(cg_ap_m1['img_celeba'],
                                                                 ap_cond_img['img_celeba']);
                    cg_ap_m1['text'] = add_mean_all_labels(cg_ap_m1['text'],
                                                           ap_cond_img['text']);
                    cg_ap_m2['img_celeba'] = add_mean_all_labels(cg_ap_m2['img_celeba'],
                                                                 ap_cond_text['img_celeba']);
                    cg_ap_m2['text'] = add_mean_all_labels(cg_ap_m2['text'],
                                                           ap_cond_text['text']);
                    coherence_random_pairs = classify_rand_gen_samples(flags,
                                                                       epoch,
                                                                       model_dict,
                                                                       rand_gen_samples);
                    random_coherence = add_mean_all_labels(random_coherence,
                                                            coherence_random_pairs);

            if train:
                if iteration == (num_batches_epoch - 1):
                    clf_lr = train_clfs_latent_representation(data, labels);
            else:
                if clf_lr is not None:
                    ap = classify_latent_representations(flags, epoch, clf_lr,
                                                         data,
                                                         labels)[-1];
                    ap_img = ap['img'];
                    ap_text = ap['text'];
                    ap_joint = ap['joint'];
                    ap_img_s = ap['style_img'];
                    ap_text_s = ap['style_text'];
                    lr_ap_m1 = add_mean_all_labels(lr_ap_m1, ap_img);
                    lr_ap_m2 = add_mean_all_labels(lr_ap_m2, ap_text);
                    lr_ap_joint = add_mean_all_labels(lr_ap_joint, ap_joint);
                    lr_ap_m1_s = add_mean_all_labels(lr_ap_m1_s, ap_img_s);
                    lr_ap_m2_s = add_mean_all_labels(lr_ap_m2_s, ap_text_s);

        # backprop
        if train == True:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            utils.printProgressBar(step_print_progress, num_batches_epoch)
        
        # write scalars to tensorboard
        name = "train" if train else "test"
        writer.add_scalars('%s/Loss' % name, {'loss': total_loss.item()}, step_logs)
        writer.add_scalars('%s/KLD' % name, {
            'Content_M1': kld_m1_class.item(),
            'Style_M1': kld_m1_style.item(),
            'Content_M2': kld_m2_class.item(),
            'Style_M2': kld_m2_style.item(),
        }, step_logs)
        writer.add_scalars('%s/KLD_individual_divs' % name, {
            'M0': results_joint['individual_divs'][0],
            'M1': results_joint['individual_divs'][1],
            'M2': results_joint['individual_divs'][2],
        }, step_logs)
        writer.add_scalars('%s/RecLoss' % name, {
            'M1': rec_error_m1.item(),
            'M2': rec_error_m2.item(),
        }, step_logs)
        writer.add_scalars('%s/mu' % name, {
            'content_group': group_mu.mean().item(),
        }, step_logs)
        writer.add_scalars('%s/logvar' % name, {
            'content_alpha': group_logvar.mean().item(),
        }, step_logs)
        writer.add_scalars('%s/group_divergence' % name, {
            'KLDgroup': kld_group.item(),
            'group_div': group_divergence.item(),
        }, step_logs)
        if flags.modality_jsd:
            writer.add_scalars('%s/group_divergence' % name, {
                'KLDdynprior': kld_dyn_prior.item(),
            }, step_logs)
        writer.add_scalars('%s/mu' % name, {
            'content_m1': m1_class_mu.mean().item(),
            'style_m1': m1_style_mu.mean().item(),
            'content_m2': m2_class_mu.mean().item(),
            'style_m2': m2_style_mu.mean().item(),
            'content_fused': group_mu.mean().item(),
        }, step_logs)
        writer.add_scalars('%s/logvar' % name, {
            'style_m1': m1_style_logvar.mean().item(),
            'content_m1': m1_class_logvar.mean().item(),
            'style_m2': m2_style_logvar.mean().item(),
            'content_m2': m2_class_logvar.mean().item(),
            'content_fused': group_logvar.mean().item(),
        }, step_logs)
        step_logs += 1
        step_print_progress += 1;

    print('')

    # write style-transfer ("swapping") figure to tensorboard
    if train == False:
        random_plots = generate_random_samples_plot(flags, epoch, vae_bimodal,
                                                    alphabet);
        random_img = random_plots['img_celeba'];
        random_text = random_plots['text'];
        writer.add_image('Random_Img', random_img, epoch, dataformats="HWC")
        writer.add_image('Random_Text', random_text, epoch, dataformats="HWC")

        swapping_figs = generate_swapping_plot(flags, epoch, vae_bimodal,
                                               test_samples, alphabet);
        swaps_img_content = swapping_figs['img_celeba'];
        swaps_text_content = swapping_figs['text'];
        swap_img_img = swaps_img_content['img_celeba'];
        swap_text_img = swaps_text_content['img_celeba'];
        writer.add_image('Swapping_Img_to_Img', swap_img_img, epoch, dataformats="HWC")
        writer.add_image('Swapping_Text_to_Img', swap_text_img, epoch, dataformats="HWC")

        cond_figs = generate_conditional_fig(flags, epoch, vae_bimodal,
                                             test_samples, alphabet)
        cond_img = cond_figs['img_celeba'];
        cond_text = cond_figs['text'];
        cond_img_img = cond_img['img_celeba']
        cond_img_text = cond_img['text'];
        cond_text_img = cond_text['img_celeba'];
        cond_text_text = cond_text['text'];
        writer.add_image('Conditional_Generation_Text_to_Img', cond_text_img, epoch, dataformats="HWC")
        writer.add_image('Conditional_Generation_Img_to_Img', cond_img_img, epoch, dataformats="HWC")
        writer.add_image('Conditional_Generation_Text_to_Text', cond_text_text, epoch, dataformats="HWC")
        writer.add_image('Conditional_Generation_Img_to_Text', cond_img_text, epoch, dataformats="HWC")
        if (epoch + 1) % flags.eval_freq == 0 or (epoch + 1) == flags.end_epoch:
            # calc diversity/variability scores
            if (epoch+1) == flags.eval_freq:
                paths = {'real': flags.dir_gen_eval_fid_real,
                         'conditional': flags.dir_gen_eval_fid_cond_gen,
                         'random': flags.dir_gen_eval_fid_random}
            else:
                paths = {'conditional': flags.dir_gen_eval_fid_cond_gen,
                         'random': flags.dir_gen_eval_fid_random}

            calculate_inception_features_for_gen_evaluation(flags, paths, modality='img_celeba');
            act_celeba = load_inception_activations(flags, 'img_celeba');
            [act_inc_real_celeba, act_inc_rand_celeba, act_inc_cond_celeba] = act_celeba;
            fid_random = calculate_fid(act_inc_real_celeba, act_inc_rand_celeba);
            fid_cond = calculate_fid(act_inc_real_celeba, act_inc_cond_celeba);
            ap_prd_random = calculate_prd(act_inc_real_celeba, act_inc_rand_celeba);
            ap_prd_cond = calculate_prd(act_inc_real_celeba, act_inc_cond_celeba);

            name_quality = 'Sample_Quality'
            writer.add_scalars('%s/fid' % name_quality, {
               'fid_random': fid_random,
               'fid_conditional': fid_cond
            }, step_logs)
            writer.add_scalars('%s/prd' % name_quality, {
                'ap_random': ap_prd_random,
                'ap_conditional': ap_prd_cond
            }, step_logs)

            lr_ap_m1 = get_mean_all_labels(lr_ap_m1);
            lr_ap_m2 = get_mean_all_labels(lr_ap_m2);
            lr_ap_joint = get_mean_all_labels(lr_ap_joint);
            lr_ap_m1_s = get_mean_all_labels(lr_ap_m1_s);
            lr_ap_m2_s = get_mean_all_labels(lr_ap_m2_s);
            cg_ap_m1['img_celeba'] = get_mean_all_labels(cg_ap_m1['img_celeba'])
            cg_ap_m1['text'] = get_mean_all_labels(cg_ap_m1['text'])
            cg_ap_m2['img_celeba'] = get_mean_all_labels(cg_ap_m2['img_celeba'])
            cg_ap_m2['text'] = get_mean_all_labels(cg_ap_m2['text'])
            random_coherence = get_mean_all_labels(random_coherence);

            name_latents = 'Representations';
            writer.add_scalars('%s/ap_mean' % name,
                               {
                                'img': np.mean(np.array(list(lr_ap_m1.values()))),
                                'text': np.mean(np.array(list(lr_ap_m2.values()))),
                               }, step_logs)
            writer.add_scalars('%s/ap_joint' % name_latents,
                               lr_ap_joint,
                               step_logs)
            writer.add_scalars('%s/ap_img' % name_latents,
                               lr_ap_m1,
                               step_logs)
            writer.add_scalars('%s/ap_img_s' % name_latents,
                               lr_ap_m1_s,
                               step_logs)
            writer.add_scalars('%s/ap_text' % name_latents,
                               lr_ap_m2,
                               step_logs)
            writer.add_scalars('%s/ap_text_s' % name_latents,
                               lr_ap_m2_s,
                               step_logs)

            name_gen = 'Generation';
            writer.add_scalars('%s/ap_mean' % name_gen,
                               {
                                'img_img': np.mean(np.array(list(cg_ap_m1['img_celeba'].values()))),
                                'img_text': np.mean(np.array(list(cg_ap_m1['text'].values()))),
                                'text_img': np.mean(np.array(list(cg_ap_m2['img_celeba'].values()))),
                                'text_text': np.mean(np.array(list(cg_ap_m2['text'].values()))),
                               }, step_logs)

            writer.add_scalars('%s/ap_img_img' % name_gen,
                               cg_ap_m1['img_celeba'],
                               step_logs)
            writer.add_scalars('%s/ap_img_text' % name_gen,
                               cg_ap_m1['text'],
                               step_logs)
            writer.add_scalars('%s/ap_text_img' % name_gen,
                               cg_ap_m2['img_celeba'],
                               step_logs)
            writer.add_scalars('%s/ap_text_text' % name_gen,
                               cg_ap_m2['text'],
                               step_logs)
            writer.add_scalars('%s/random_coherence' % name_gen,
                               random_coherence,
                               step_logs)
            writer.add_scalars('%s/random_coherence_mean' % name_gen,
                               {
                                'mean': np.mean(np.mean(list(random_coherence.values())))},
                               step_logs)
    return step_logs, clf_lr, writer;


