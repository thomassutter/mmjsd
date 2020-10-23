import os

import torch
import torch.nn as nn

from networks.ConvNetworksImgSVHN import EncoderSVHN, DecoderSVHN
from networks.ConvNetworksImgMNIST import EncoderImg, DecoderImg
from networks.ConvNetworksTextMNIST import EncoderText, DecoderText
from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_poe
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import calc_kl_divergence
from divergence_measures.mm_div import poe

from utils import utils


class VAEtrimodalSVHNMNIST(nn.Module):
    def __init__(self, flags):
        super(VAEtrimodalSVHNMNIST, self).__init__()
        self.num_modalities = 3;
        self.flags = flags;
        self.encoder_svhn = EncoderSVHN(flags)
        self.encoder_mnist = EncoderImg(flags)
        self.encoder_text = EncoderText(flags)
        self.decoder_mnist = DecoderImg(flags);
        self.decoder_svhn = DecoderSVHN(flags);
        self.decoder_text = DecoderText(flags);
        self.encoder_mnist = self.encoder_mnist.to(flags.device);
        self.encoder_svhn = self.encoder_svhn.to(flags.device);
        self.encoder_text = self.encoder_text.to(flags.device);
        self.decoder_mnist = self.decoder_mnist.to(flags.device);
        self.decoder_svhn = self.decoder_svhn.to(flags.device);
        self.decoder_text = self.decoder_text.to(flags.device);
        self.lhood_mnist = utils.get_likelihood(flags.likelihood_m1);
        self.lhood_svhn = utils.get_likelihood(flags.likelihood_m2);
        self.lhood_text = utils.get_likelihood(flags.likelihood_m3);

        d_size_m1 = flags.img_size_mnist*flags.img_size_mnist;
        d_size_m2 = 3*flags.img_size_svhn*flags.img_size_svhn;
        d_size_m3 = flags.len_sequence;
        total_d_size = d_size_m1 + d_size_m2 + d_size_m3;
        w1 = d_size_m2/d_size_m1;
        w2 = 1.0;
        w3 = d_size_m2/d_size_m3;
        w_total = w1+w2+w3;
        self.rec_w1 = w1;
        self.rec_w2 = w2;
        self.rec_w3 = w3;

        weights = utils.reweight_weights(torch.Tensor(flags.alpha_modalities));
        self.weights = weights.to(flags.device);
        if flags.modality_moe or flags.modality_jsd:
            self.modality_fusion = self.moe_fusion;
            self.calc_joint_divergence = self.divergence_moe;
            if flags.modality_jsd:
                self.calc_joint_divergence = self.divergence_jsd;
        elif flags.modality_poe:
            self.modality_fusion = self.poe_fusion;
            self.calc_joint_divergence = self.divergence_poe;


    def forward(self, input_mnist=None, input_svhn=None, input_text=None):
        latents = self.inference(input_mnist, input_svhn, input_text);
        results = dict();
        results['latents'] = latents;

        results['group_distr'] = latents['joint'];
        class_embeddings = utils.reparameterize(latents['joint'][0],
                                                latents['joint'][1]);
        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights']);
        for k, key in enumerate(div.keys()):
            results[key] = div[key];

        results_rec = dict();
        if input_mnist is not None:
            m1_s_mu, m1_s_logvar = latents['img_mnist'][:2];
            if self.flags.factorized_representation:
                m1_s_embeddings = utils.reparameterize(mu=m1_s_mu, logvar=m1_s_logvar);
            else:
                m1_s_embeddings = None;
            m1_rec = self.lhood_mnist(*self.decoder_mnist(m1_s_embeddings, class_embeddings));
            results_rec['img_mnist'] = m1_rec;
        if input_svhn is not None:
            m2_s_mu, m2_s_logvar = latents['img_svhn'][:2];
            if self.flags.factorized_representation:
                m2_s_embeddings = utils.reparameterize(mu=m2_s_mu, logvar=m2_s_logvar);
            else:
                m2_s_embeddings = None;
            m2_rec = self.lhood_svhn(*self.decoder_svhn(m2_s_embeddings, class_embeddings));
            results_rec['img_svhn'] = m2_rec;
        if input_text is not None:
            m3_s_mu, m3_s_logvar = latents['text'][:2];
            if self.flags.factorized_representation:
                m3_s_embeddings = utils.reparameterize(mu=m3_s_mu, logvar=m3_s_logvar);
            else:
                m3_s_embeddings = None;
            m3_rec = self.lhood_text(*self.decoder_text(m3_s_embeddings, class_embeddings));
            results_rec['text'] = m3_rec;
        results['rec'] = results_rec;
        return results;


    def divergence_poe(self, mus, logvars, weights=None):
        div_measures = calc_group_divergence_poe(self.flags,
                                         mus,
                                         logvars,
                                         norm=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0];
        divs['individual_divs'] = div_measures[1];
        divs['dyn_prior'] = None;
        return divs;


    def divergence_moe(self, mus, logvars, weights=None):
        if weights is None:
            weights=self.weights;
        weights = weights.clone();
        weights[0] = 0.0;
        weights = utils.reweight_weights(weights);
        div_measures = calc_group_divergence_moe(self.flags,
                                                 mus,
                                                 logvars,
                                                 weights,
                                                 normalization=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0];
        divs['individual_divs'] = div_measures[1];
        divs['dyn_prior'] = None;
        return divs;


    def divergence_jsd(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights;
        div_measures = calc_alphaJSD_modalities(self.flags,
                                                mus,
                                                logvars,
                                                weights,
                                                normalization=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0];
        divs['individual_divs'] = div_measures[1];
        divs['dyn_prior'] = div_measures[2];
        return divs;



    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights;

        weights[0] = 0.0;
        weights = utils.reweight_weights(weights);
        num_samples = mus[0].shape[0];
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
                                                               mus,
                                                               logvars,
                                                               weights,
                                                               num_samples);
        return [mu_moe, logvar_moe];


    def poe_fusion(self, mus, logvars, weights=None):
        mu_poe, logvar_poe = poe(mus, logvars);
        return [mu_poe, logvar_poe];


    def encode(self, i_mnist=None, i_svhn=None, i_text=None):
        latents = dict();
        if i_mnist is not None:
            latents['img_mnist'] = self.encoder_mnist(i_mnist)
        else:
            latents['img_mnist'] = [None, None, None, None];
        if i_svhn is not None:
            latents['img_svhn'] = self.encoder_svhn(i_svhn);
        else:
            latents['img_svhn'] = [None, None, None, None];
        if i_text is not None:
            latents['text'] = self.encoder_text(i_text);
        else:
            latents['text'] = [None, None, None, None];
        return latents;


    def inference(self, input_mnist=None, input_svhn=None, input_text=None):
        latents = self.encode(i_mnist=input_mnist,
                              i_svhn=input_svhn,
                              i_text=input_text);
        mod_avail = 0;
        if input_mnist is not None:
            num_samples = input_mnist.shape[0];
            mod_avail += 1;
        if input_svhn is not None:
            num_samples = input_svhn.shape[0];
            mod_avail += 1;
        if input_text is not None:
            num_samples = input_text.shape[0];
            mod_avail += 1;
        mus = torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device);
        logvars = torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device);
        weights = [self.weights[0]];
        if input_mnist is not None:
            mnist_mu, mnist_logvar = latents['img_mnist'][2:];
            mus = torch.cat([mus, mnist_mu.unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, mnist_logvar.unsqueeze(0)], dim=0);
            weights.append(self.weights[1]);
        if input_svhn is not None:
            svhn_mu, svhn_logvar = latents['img_svhn'][2:];
            mus = torch.cat([mus, svhn_mu.unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, svhn_logvar.unsqueeze(0)], dim=0);
            weights.append(self.weights[2]);
        if input_text is not None:
            text_mu, text_logvar = latents['text'][2:];
            mus = torch.cat([mus, text_mu.unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars, text_logvar.unsqueeze(0)], dim=0);
            weights.append(self.weights[3]);
        weights = torch.Tensor(weights).to(self.flags.device)
        joint_mu, joint_logvar = self.modality_fusion(mus, logvars,
                                                      weights.clone());
        latents['mus'] = mus; latents['logvars'] = logvars;
        latents['weights'] = weights;
        latents['joint'] = [joint_mu, joint_logvar];
        return latents;


    def get_random_styles(self, num_samples):
        if self.flags.factorized_representation:
            z_style_1 = torch.randn(num_samples, self.flags.style_mnist_dim);
            z_style_2 = torch.randn(num_samples, self.flags.style_svhn_dim);
            z_style_3 = torch.randn(num_samples, self.flags.style_text_dim);
            z_style_1 = z_style_1.to(self.flags.device);
            z_style_2 = z_style_2.to(self.flags.device);
            z_style_3 = z_style_3.to(self.flags.device);
        else:
            z_style_1 = None;
            z_style_2 = None;
            z_style_3 = None;
        styles = {'img_mnist': z_style_1, 'img_svhn': z_style_2, 'text': z_style_3};
        return styles;


    def get_random_style_dists(self, num_samples):
        s1_mu = torch.zeros(num_samples,
                            self.flags.style_mnist_dim).to(self.flags.device)
        s1_logvar = torch.zeros(num_samples,
                                self.flags.style_mnist_dim).to(self.flags.device);
        s2_mu = torch.zeros(num_samples,
                            self.flags.style_svhn_dim).to(self.flags.device)
        s2_logvar = torch.zeros(num_samples,
                                self.flags.style_svhn_dim).to(self.flags.device);
        s3_mu = torch.zeros(num_samples,
                            self.flags.style_text_dim).to(self.flags.device)
        s3_logvar = torch.zeros(num_samples,
                                self.flags.style_text_dim).to(self.flags.device);
        m1_dist = [s1_mu, s1_logvar];
        m2_dist = [s2_mu, s2_logvar];
        m3_dist = [s3_mu, s3_logvar];
        styles = {'img_mnist': m1_dist, 'img_svhn': m2_dist, 'text': m3_dist};
        return styles;


    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;
        z_class = torch.randn(num_samples, self.flags.class_dim);
        z_class = z_class.to(self.flags.device);

        style_latents = self.get_random_styles(num_samples);
        random_latents = {'content': z_class, 'style': style_latents};
        random_samples = self.generate_from_latents(random_latents);
        return random_samples;


    def generate_from_latents(self, latents):
        suff_stats = self.generate_sufficient_statistics_from_latents(latents);
        cond_gen_mnist = suff_stats['img_mnist'].mean;
        cond_gen_svhn = suff_stats['img_svhn'].mean;
        cond_gen_text = suff_stats['text'].mean;
        cond_gen = {'img_mnist': cond_gen_mnist,
                    'img_svhn': cond_gen_svhn,
                    'text': cond_gen_text};
        return cond_gen;


    def generate_sufficient_statistics_from_latents(self, latents):
        style_mnist = latents['style']['img_mnist'];
        style_svhn = latents['style']['img_svhn'];
        style_text = latents['style']['text'];
        content = latents['content']
        cond_gen_m1 = self.lhood_mnist(*self.decoder_mnist(style_mnist, content));
        cond_gen_m2 = self.lhood_svhn(*self.decoder_svhn(style_svhn, content));
        cond_gen_m3 = self.lhood_text(*self.decoder_text(style_text, content));
        return {'img_mnist': cond_gen_m1, 'img_svhn': cond_gen_m2, 'text': cond_gen_m3}


    def cond_generation_1a(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;

        style_latents = self.get_random_styles(num_samples);
        cond_gen_samples = dict();
        for k, key in enumerate(latent_distributions):
            [mu, logvar] = latent_distributions[key];
            content_rep = utils.reparameterize(mu=mu, logvar=logvar);
            latents = {'content': content_rep, 'style': style_latents}
            cond_gen_samples[key] = self.generate_from_latents(latents);
        return cond_gen_samples;


    def cond_generation_2a(self, latent_distribution_pairs, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;
        
        mu0 = torch.zeros(1, num_samples, self.flags.class_dim);
        logvar0 = torch.zeros(1, num_samples, self.flags.class_dim);
        mu0 = mu0.to(self.flags.device);
        logvar0 = logvar0.to(self.flags.device);
        style_latents = self.get_random_styles(num_samples);
        cond_gen_2a = dict();
        for p, pair in enumerate(latent_distribution_pairs.keys()):
            ld_pair = latent_distribution_pairs[pair];
            mu_list = [mu0]; logvar_list = [logvar0];
            for k, key in enumerate(ld_pair['latents'].keys()):
                mu_list.append(ld_pair['latents'][key][0].unsqueeze(0));
                logvar_list.append(ld_pair['latents'][key][1].unsqueeze(0));
            mus = torch.cat(mu_list, dim=0);
            logvars = torch.cat(logvar_list, dim=0);
            weights_pair = ld_pair['weights']
            weights_pair.insert(0, self.weights[0])
            weights_pair = utils.reweight_weights(torch.Tensor(weights_pair));
            mu_joint, logvar_joint = self.modality_fusion(mus, logvars, weights_pair)
            #mu_joint, logvar_joint = poe(mus, logvars);
            c_emb = utils.reparameterize(mu_joint, logvar_joint);
            l_2a = {'content': c_emb, 'style': style_latents};
            cond_gen_2a[pair] = self.generate_from_latents(l_2a);
        return cond_gen_2a;


    def save_networks(self):
        torch.save(self.encoder_mnist.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m1))
        torch.save(self.decoder_mnist.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m1))
        torch.save(self.encoder_svhn.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m2))
        torch.save(self.decoder_svhn.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m2))
        torch.save(self.encoder_text.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m3))
        torch.save(self.decoder_text.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m3))
