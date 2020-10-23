import os

import torch
import torch.nn as nn

from networks.ConvNetworksImgCelebA import EncoderImg, DecoderImg
from networks.ConvNetworksTextCelebA import EncoderText, DecoderText
from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_poe
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import calc_kl_divergence
from divergence_measures.mm_div import poe
from utils import utils


class VAEbimodalCelebA(nn.Module):
    def __init__(self, flags):
        super(VAEbimodalCelebA, self).__init__();
        self.flags = flags;
        self.encoder_img = EncoderImg(flags)
        self.encoder_text = EncoderText(flags)
        self.decoder_img = DecoderImg(flags);
        self.decoder_text = DecoderText(flags);
        self.lhood_celeba = utils.get_likelihood(flags.likelihood_m1);
        self.lhood_text = utils.get_likelihood(flags.likelihood_m2);
        self.encoder_img = self.encoder_img.to(flags.device);
        self.decoder_img = self.decoder_img.to(flags.device);
        self.encoder_text = self.encoder_text.to(flags.device);
        self.decoder_text = self.decoder_text.to(flags.device);

        d_size_m1 = flags.img_size*flags.img_size;
        d_size_m2 = flags.len_sequence;
        w1 = 1.0;
        w2 = d_size_m1/d_size_m2;
        w_total = w1+w2;
        #w1 = w1/w_total;
        #w2 = w2/w_total;
        self.rec_w1 = w1;
        self.rec_w2 = w2;

        weights = utils.reweight_weights(torch.Tensor(flags.alpha_modalities));
        self.weights = weights.to(flags.device);
        if flags.modality_moe or flags.modality_jsd:
            self.modality_fusion = self.moe_fusion;
            if flags.modality_moe:
                self.calc_joint_divergence = self.divergence_moe;
            if flags.modality_jsd:
                self.calc_joint_divergence = self.divergence_jsd;
        elif flags.modality_poe:
            self.modality_fusion = self.poe_fusion;
            self.calc_joint_divergence = self.divergence_poe;


    def forward(self, input_img=None, input_text=None):
        results = dict();
        latents = self.inference(input_img=input_img, input_text=input_text);
        results['latents'] = latents;
        mus = latents['mus']
        logvars = latents['logvars']
        weights = latents['weights']
        if input_img is not None and input_text is not None:
            div = self.calc_joint_divergence(mus, logvars, weights);
            for k,key in enumerate(div.keys()):
                results[key] = div[key];

        results['group_distr'] = latents['joint'];
        class_embeddings = utils.reparameterize(latents['joint'][0],
                                                latents['joint'][1])

        if self.flags.factorized_representation:
            if input_img is not None:
                [m1_s_mu, m1_s_logvar] = latents['img_celeba'][:2];
                m1_style_latent_embeddings = utils.reparameterize(mu=m1_s_mu, logvar=m1_s_logvar)
            if input_text is not None:
                [m2_s_mu, m2_s_logvar] = latents['text'][:2];
                m2_style_latent_embeddings = utils.reparameterize(mu=m2_s_mu, logvar=m2_s_logvar)
        else:
            m1_style_latent_embeddings = None;
            m2_style_latent_embeddings = None;

        m1_rec = None;
        m2_rec = None;
        if input_img is not None:
            m1_rec = self.lhood_celeba(*self.decoder_img(m1_style_latent_embeddings, class_embeddings));
        if input_text is not None:
            m2_rec = self.lhood_text(*self.decoder_text(m2_style_latent_embeddings, class_embeddings));
        results['rec'] = {'img_celeba': m1_rec, 'text': m2_rec};
        return results;


    def generate(self, num_samples):
        z_class = torch.randn(num_samples, self.flags.class_dim);
        z_class = z_class.to(self.flags.device);
        if self.flags.factorized_representation:
            z_style_1 = torch.randn(num_samples, self.flags.style_m1_dim);
            z_style_2 = torch.randn(num_samples, self.flags.style_m2_dim);
            z_style_1 = z_style_1.to(self.flags.device);
            z_style_2 = z_style_2.to(self.flags.device);
        else:
            z_style_1 = None;
            z_style_2 = None;
        style_latents = {'img_celeba': z_style_1, 'text': z_style_2}
        random_latents = {'content': z_class, 'style': style_latents};
        random_samples = self.generate_from_latents(random_latents);
        return random_samples;


    def generate_from_latents(self, latents):
        style_img = latents['style']['img_celeba'];
        style_text = latents['style']['text'];
        content = latents['content']
        if self.lhood_celeba is not None and self.lhood_text is not None:
            if style_img is not None:
                cond_gen_m1 = self.lhood_celeba(*self.decoder_img(style_img, content));
                cond_gen_m1 = cond_gen_m1.mean;
            else:
                cond_gen_m1 = None;
            if style_text is not None:
                cond_gen_m2 = self.lhood_text(*self.decoder_text(style_text, content));
                cond_gen_m2 = cond_gen_m2.mean;
            else:
                cond_gen_m2 = None;
        else:
            cond_gen_m1 = self.decoder_img(style_img, content)[0];
            cond_gen_m2 = self.decoder_text(style_text, content)[0];
        cond_gen = {'img_celeba': cond_gen_m1, 'text': cond_gen_m2};
        return cond_gen;

    def generate_sufficient_statistics_from_latents(self, latents):
        style_img = latents['style']['img_celeba'];
        style_text = latents['style']['text'];
        content = latents['content']
        cond_gen_m1 = self.lhood_celeba(*self.decoder_img(style_img, content));
        cond_gen_m2 = self.lhood_text(*self.decoder_text(style_text, content));
        return {'img_celeba': cond_gen_m1, 'text': cond_gen_m2}


    def cond_generation(self, latent_distributions):
        if 'img_celeba' in latent_distributions:
            [m1_mu, m1_logvar] = latent_distributions['img_celeba'];
            content_cond_m1 = utils.reparameterize(mu=m1_mu, logvar=m1_logvar);
            num_samples = m1_mu.shape[0];
        if 'text' in latent_distributions:
            [m2_mu, m2_logvar] = latent_distributions['text'];
            content_cond_m2 = utils.reparameterize(mu=m2_mu, logvar=m2_logvar);
            num_samples = m2_mu.shape[0];

        if self.flags.factorized_representation:
            random_style_m1 = torch.randn(num_samples, self.flags.style_m1_dim);
            random_style_m2 = torch.randn(num_samples, self.flags.style_m2_dim);
            random_style_m1 = random_style_m1.to(self.flags.device)
            random_style_m2 = random_style_m2.to(self.flags.device)
        else:
            random_style_m1 = None;
            random_style_m2 = None;

        style_latents = {'img_celeba': random_style_m1, 'text': random_style_m2};
        cond_gen_samples = dict();
        if 'img_celeba' in latent_distributions:
            latents_mnist = {'content': content_cond_m1,
                             'style': style_latents}
            cond_gen_samples['img_celeba'] = self.generate_from_latents(latents_mnist);
        if 'text' in latent_distributions:
            latents_svhn = {'content': content_cond_m2,
                             'style': style_latents}
            cond_gen_samples['text'] = self.generate_from_latents(latents_svhn);
        return cond_gen_samples;


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
            weights = self.weights;
        weights = weights.clone()
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
        weights = weights.clone();
        weights = utils.reweight_weights(weights);
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
        weights = weights.clone();
        weights[0] = 0.0;
        weights = utils.reweight_weights(weights);
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
                                                               mus,
                                                               logvars,
                                                               weights);
        return [mu_moe, logvar_moe];


    def poe_fusion(self, mus, logvars, weights=None):
        mu_poe, logvar_poe = poe(mus, logvars);
        return [mu_poe, logvar_poe];


    def encode(self, i_img=None, i_text=None):
        latents = dict();
        if i_img is not None:
            m1_style_mu, m1_style_logvar, m1_class_mu, m1_class_logvar = self.encoder_img(i_img)
            latents['img_celeba'] = [m1_style_mu, m1_style_logvar, m1_class_mu, m1_class_logvar];
        else:
            latents['img_celeba'] = None;

        if i_text is not None:
            m2_style_mu, m2_style_logvar, m2_class_mu, m2_class_logvar = self.encoder_text(i_text);
            latents['text'] = [m2_style_mu, m2_style_logvar, m2_class_mu, m2_class_logvar];
        else:
            latents['text'] = None;
        return latents;


    def inference(self, input_img=None, input_text=None):
        latents = self.encode(input_img, input_text);
        if input_img is not None:
            num_samples = input_img.shape[0];
        if input_text is not None:
            num_samples = input_text.shape[0];

        mus = torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device);
        logvars = torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device);
        weights = [self.weights[0]]
        if input_img is not None:
            mus = torch.cat([mus,
                             latents['img_celeba'][2].unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars,
                                 latents['img_celeba'][3].unsqueeze(0)], dim=0);
            weights.append(self.weights[1]);
        if input_text is not None:
            mus = torch.cat([mus,
                             latents['text'][2].unsqueeze(0)], dim=0);
            logvars = torch.cat([logvars,
                                 latents['text'][3].unsqueeze(0)], dim=0);
            weights.append(self.weights[2]);
        
        weights = torch.Tensor(weights).to(self.flags.device);
        joint_mu, joint_logvar = self.modality_fusion(mus, logvars, weights);
        latents['joint'] = [joint_mu, joint_logvar];
        latents['mus'] = mus;
        latents['logvars'] = logvars;
        latents['weights'] = weights;
        return latents;


    def encode_img(self, img):
        m1_style_mu, m1_style_logvar, m1_class_mu, m1_class_logvar = self.encoder_img(img);
        return [m1_style_mu, m1_style_logvar, m1_class_mu, m1_class_logvar];


    def encode_text(self, text):
        m2_style_mu, m2_style_logvar, m2_class_mu, m2_class_logvar = self.encoder_text(text)
        return [m2_style_mu, m2_style_logvar, m2_class_mu, m2_class_logvar];


    def decode_img(self, emb_style, emb_class):
        reconstruction = self.decoder_img(emb_style, emb_class);
        return reconstruction;


    def decode_text(self, emb_style, emb_class):
        reconstruction = self.decoder_text(emb_style, emb_class);
        return reconstruction;


    def save_networks(self):
        torch.save(self.encoder_img.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m1))
        torch.save(self.decoder_img.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m1))
        torch.save(self.encoder_text.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m2))
        torch.save(self.decoder_text.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m2))


