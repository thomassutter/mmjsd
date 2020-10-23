
import numpy as np
from scipy.special import logsumexp
from itertools import cycle
import math

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import utils
from utils.constants_svhnmnist import indices as IND3
from divergence_measures.mm_div import alpha_poe
from divergence_measures.mm_div import poe

LOG2PI = float(np.log(2.0 * math.pi))


def get_synergy_dist(flags, flows, num_imp_samples):
    flow_reps = torch.zeros(1, flags.batch_size*num_imp_samples, flags.class_dim);
    flow_reps = flow_reps.to(flags.device);
    for k, key in enumerate(flows.keys()):
        flow_reps = torch.cat([flow_reps, flows[key]['content'][2].unsqueeze(0)]);
    # only works if modalities are equally weighted
    weights_mixture_selection = utils.reweight_weights(torch.Tensor([0.0, flags.alpha_modalities[1],
                                                                     flags.alpha_modalities[2]]));
    weights_mixture_selection = weights_mixture_selection.to(flags.device);
    flow_emb_moe = utils.flow_mixture_component_selection(flags, flow_reps, weights_mixture_selection,
                                                          num_samples=flags.batch_size*num_imp_samples);
    return flow_emb_moe;


def get_latent_samples(flags, latents, mod_names):
    l_c = latents['content'];
    l_s = latents['style'];
    c_emb = utils.reparameterize(l_c[0], l_c[1]);
    styles = dict();
    c = {'mu': l_c[0], 'logvar': l_c[1], 'z': c_emb}
    if flags.factorized_representation:
        for k, key in enumerate(l_s.keys()):
            s_emb = utils.reparameterize(l_s[key][0], l_s[key][1]);
            s = {'mu': l_s[key][0], 'logvar': l_s[key][1], 'z': s_emb}
            styles[key] = s;
    else:
        for k, key in enumerate(mod_names):
            styles[key] = None;
    emb = {'content': c, 'style': styles}
    return emb;


def get_dyn_prior(weights, mus, logvars):
    mu_poe, logvar_poe = alpha_poe(weights, mus, logvars);
    return [mu_poe, logvar_poe];


#at the moment: only marginals and joint
def calc_log_likelihood_batch(flags, mod, batch, model, mod_weights, num_imp_samples=10):
    mnist_batch, svhn_batch, text_batch, labels_batch = batch;
    num_samples_batch, m1, m2, m3 = mnist_batch.shape;
    num_samples_batch, s1, s2, s3 = svhn_batch.shape;
    num_samples_batch, t1, t2 = text_batch.shape;
    #TODO: add permutation of samples in batch
    num_samples = num_samples_batch*num_imp_samples;
    mnist_batch = mnist_batch.unsqueeze(0).repeat(num_imp_samples, 1, 1, 1, 1);
    svhn_batch = svhn_batch.unsqueeze(0).repeat(num_imp_samples, 1, 1, 1, 1);
    text_batch = text_batch.unsqueeze(0).repeat(num_imp_samples, 1, 1, 1);
    mnist_batch = mnist_batch.view(num_samples, m1, m2, m3);
    svhn_batch = svhn_batch.view(num_samples, s1, s2, s3);
    text_batch = text_batch.view(num_samples, t1, t2);
    mnist_batch = mnist_batch.to(flags.device);
    svhn_batch = svhn_batch.to(flags.device);
    text_batch = text_batch.to(flags.device);
    batch_joint = {'img_mnist': mnist_batch, 'img_svhn': svhn_batch, 'text': text_batch}
    if mod == 'img_mnist':
        i_mnist = mnist_batch;
        i_svhn = None;
        i_text = None;
    elif mod == 'img_svhn':
        i_mnist = None;
        i_svhn = svhn_batch;
        i_text = None;
    elif mod == 'text':
        i_mnist = None;
        i_svhn = None;
        i_text = text_batch;
    elif mod == 'mnist_svhn' or mod == 'mnist_svhn_dp':
        i_mnist = mnist_batch;
        i_svhn = svhn_batch;
        i_text = None;
    elif mod == 'svhn_text' or mod == 'svhn_text_dp':
        i_mnist = None;
        i_svhn = svhn_batch;
        i_text = text_batch;
    elif mod == 'mnist_text' or mod == 'mnist_text_dp':
        i_mnist = mnist_batch;
        i_svhn = None;
        i_text = text_batch;
    elif mod == 'joint':
        i_mnist = mnist_batch;
        i_svhn = svhn_batch;
        i_text = text_batch;

    if model.num_modalities == 2:
        mod_names = IND2.keys();
        latents = model.inference(input_mnist=i_mnist, input_svhn=i_svhn);
    else:
        mod_names = IND3.keys();
        latents = model.inference(input_mnist=i_mnist,
                                  input_svhn=i_svhn,
                                  input_text=i_text);

    c_mu, c_logvar = latents['joint'];
    if mod.endswith('dp'):
        c_mu, c_logvar = poe(latents['mus'], latents['logvars']);
    style = dict();
    random_styles = model.get_random_style_dists(flags.batch_size*num_imp_samples);
    if flags.factorized_representation:
        for k, key in enumerate(mod_names):
            if latents[key][0] is None and latents[key][1] is None:
                style[key] = random_styles[key];
            else:
                style[key] = latents[key][:2];
    else:
        style = None;

    l_mod = {'content': [c_mu, c_logvar], 'style': style};
    l = get_latent_samples(flags, l_mod, mod_names);
    dyn_prior = None;
    use_mnist_style = False;
    use_svhn_style = False;
    use_text_style = False;
    if mod == 'img_mnist':
        use_mnist_style = True;
    elif mod == 'img_svhn':
        use_svhn_style = True;
    elif mod == 'text':
        use_text_style = True;
    else:
        if flags.modality_jsd:
            dyn_prior = get_dyn_prior(latents['weights'],
                                      latents['mus'],
                                      latents['logvars'])
        if mod == 'mnist_svhn' or mod == 'mnist_svhn_dp':
            use_mnist_style = True;
            use_svhn_style = True;
        elif mod == 'mnist_text' or mod == 'mnist_text_dp':
            use_mnist_style = True;
            use_text_style = True;
        elif mod == 'svhn_text' or mod == 'svhn_text_dp':
            use_svhn_style = True;
            use_text_style = True;
        elif mod == 'joint':
            use_mnist_style = True;
            use_svhn_style = True;
            use_text_style = True;

    m = l['style']['img_mnist'];
    s = l['style']['img_svhn'];
    c = l['content'];
    c_z_k = c['z'];
    if flags.factorized_representation:
        m_z_k = m['z'];
        s_z_k = s['z'];
        style_z_m = m_z_k.view(flags.batch_size * num_imp_samples, -1);
        style_z_s = s_z_k.view(flags.batch_size * num_imp_samples, -1);
    else:
        style_z_m = None;
        style_z_s = None;

    style_marg = {'img_mnist': style_z_m, 'img_svhn': style_z_s};
    if len(mod_weights) > 3:
        t = l['style']['text'];
        if flags.factorized_representation:
            style_z_t = t['z'].view(flags.batch_size*num_imp_samples, -1);
        else:
            style_z_t = None;
        style_marg = {'img_mnist': style_z_m, 'img_svhn': style_z_s, 'text': style_z_t};

    z_content = c_z_k.view(num_samples, -1);
    latents_dec = {'content': z_content, 'style': style_marg};
    gen = model.generate_sufficient_statistics_from_latents(latents_dec);
    suff_stats_mnist = gen['img_mnist'];
    suff_stats_svhn = gen['img_svhn'];

    # compute marginal log-likelihood
    if use_mnist_style:
        ll_mnist = log_marginal_estimate(flags, num_imp_samples, gen['img_mnist'], mnist_batch, m, c)
    else:
        ll_mnist = log_marginal_estimate(flags, num_imp_samples, gen['img_mnist'], mnist_batch, None, c)

    if use_svhn_style:
        ll_svhn = log_marginal_estimate(flags, num_imp_samples, gen['img_svhn'], svhn_batch, s, c)
    else:
        ll_svhn = log_marginal_estimate(flags, num_imp_samples, gen['img_svhn'], svhn_batch, None, c)
    ll = {'img_mnist': ll_mnist, 'img_svhn': ll_svhn};
    if len(mod_weights) > 3:
        if use_text_style:
            ll['text'] = log_marginal_estimate(flags, num_imp_samples, gen['text'], text_batch, t, c)
        else:
            ll['text'] = log_marginal_estimate(flags, num_imp_samples, gen['text'], text_batch, None, c)
    ll_joint = log_joint_estimate(flags, num_imp_samples, gen, batch_joint, l['style'], c);
    ll['joint'] = ll_joint;
    return ll;


def log_marginal_estimate(flags, n_samples, likelihood, image, style, content, dynamic_prior=None):
    r"""Estimate log p(x). NOTE: this is not the objective that
    should be directly optimized.
    @param ss_list: list of sufficient stats, i.e., list of
                        torch.Tensor (batch size x # samples x 784)
    @param image: torch.Tensor (batch size x 784)
                  original observed image
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param mu: torch.Tensor (batch_size x # samples x z dim)
               means of variational distribution
    @param logvar: torch.Tensor (batch_size x # samples x z dim)
                   log-variance of variational distribution
    """
    batch_size = flags.batch_size;
    if style is not None:
        z_style = style['z'];
        logvar_style = style['logvar'];
        mu_style = style['mu'];
        n, z_style_dim = z_style.size()
        style_log_q_z_given_x_2d = gaussian_log_pdf(z_style, mu_style, logvar_style);
        log_p_z_2d_style = unit_gaussian_log_pdf(z_style)

    z_content = content['z']
    mu_content = content['mu'];
    logvar_content = content['logvar'];
    log_p_x_given_z_2d = likelihood.log_prob(image).view(batch_size*n_samples,-1).sum(dim=1)
    content_log_q_z_given_x_2d = gaussian_log_pdf(z_content, mu_content, logvar_content);


    if dynamic_prior is None:
        log_p_z_2d_content = unit_gaussian_log_pdf(z_content)
    else:
        mu_prior = dynamic_prior['mu'];
        logvar_prior = dynamic_prior['logvar'];
        log_p_z_2d_content = gaussian_log_pdf(z_content, mu_prior, logvar_prior);

    if style is not None:
        log_p_z_2d = log_p_z_2d_style+log_p_z_2d_content;
        log_q_z_given_x_2d = style_log_q_z_given_x_2d + content_log_q_z_given_x_2d
    else:
        log_p_z_2d = log_p_z_2d_content;
        log_q_z_given_x_2d = content_log_q_z_given_x_2d;
    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d;
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)


def log_joint_estimate(flags, n_samples, likelihoods, targets, styles, content, dynamic_prior=None):
    r"""Estimate log p(x,y).
    @param recon_image: torch.Tensor (batch size x # samples x 784)
                        reconstructed means on bernoulli
    @param image: torch.Tensor (batch size x 784)
                  original observed image
    @param recon_label: torch.Tensor (batch_size x # samples x n_class)
                        reconstructed logits
    @param label: torch.Tensor (batch_size)
                  original observed labels
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param mu: torch.Tensor (batch_size x # samples x z dim)
               means of variational distribution
    @param logvar: torch.Tensor (batch_size x # samples x z dim)
                   log-variance of variational distribution
    """
    batch_size = flags.batch_size;
    if styles is not None:
        styles_log_q_z_given_x_2d = dict();
        styles_p_z_2d = dict();
        for key in styles.keys():
            if styles[key] is not None:
                style_m = styles[key];
                z_style_m = style_m['z'];
                logvar_style_m = style_m['logvar'];
                mu_style_m = style_m['mu'];
                style_m_log_q_z_given_x_2d = gaussian_log_pdf(z_style_m, mu_style_m, logvar_style_m);
                log_p_z_2d_style_m = unit_gaussian_log_pdf(z_style_m)
                styles_log_q_z_given_x_2d[key] = style_m_log_q_z_given_x_2d;
                styles_p_z_2d[key] = log_p_z_2d_style_m;

    z_content = content['z']
    mu_content = content['mu'];
    logvar_content = content['logvar'];

    num_mods = len(styles.keys())
    log_px_zs = torch.zeros(num_mods, batch_size * n_samples);
    log_px_zs = log_px_zs.to(flags.device);
    for k, key in enumerate(styles.keys()):
        batch = targets[key]
        lhood = likelihoods[key]
        log_p_x_given_z_2d = lhood.log_prob(batch).view(batch_size * n_samples, -1).sum(dim=1);
        log_px_zs[k] = log_p_x_given_z_2d;

    # compute components of likelihood estimate
    log_joint_zs_2d = log_px_zs.sum(0)  # sum over modalities

    if dynamic_prior is None:
        log_p_z_2d_content = unit_gaussian_log_pdf(z_content)
    else:
        mu_prior = dynamic_prior['mu'];
        logvar_prior = dynamic_prior['logvar'];
        log_p_z_2d_content = gaussian_log_pdf(z_content, mu_prior, logvar_prior);

    content_log_q_z_given_x_2d = gaussian_log_pdf(z_content, mu_content, logvar_content);
    log_p_z_2d = log_p_z_2d_content;
    log_q_z_given_x_2d = content_log_q_z_given_x_2d;
    if styles is not None:
        for k, key in enumerate(styles.keys()):
            if key in styles_p_z_2d and key in styles_log_q_z_given_x_2d:
                log_p_z_2d += styles_p_z_2d[key];
                log_q_z_given_x_2d += styles_log_q_z_given_x_2d[key];

    log_weight_2d = log_joint_zs_2d + log_p_z_2d - log_q_z_given_x_2d;
    log_weight = log_weight_2d.view(batch_size, n_samples)
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)


def log_mean_exp(x, dim=1):
    """
    log(1/k * sum(exp(x))): this normalizes x.
    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


def gaussian_log_pdf(x, mu, logvar):
    """
    Log-likelihood of data given ~N(mu, exp(logvar))
    @param x: samples from gaussian
    @param mu: mean of distribution
    @param logvar: log variance of distribution
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - logvar / 2. - torch.pow(x - mu, 2) / (2. * torch.exp(logvar))
    return torch.sum(log_pdf, dim=1)


def unit_gaussian_log_pdf(x):
    """
    Log-likelihood of data given ~N(0, 1)
    @param x: PyTorch.Tensor
              samples from gaussian
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - math.log(1.) / 2. - torch.pow(x, 2) / 2.
    return torch.sum(log_pdf, dim=1)
