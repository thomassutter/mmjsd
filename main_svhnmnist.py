import sys
import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from training.training_svhnmnist import run_epoch

from networks.VAEtrimodalSVHNMNIST import VAEtrimodalSVHNMNIST
from networks.ConvNetworkImgClfMNIST import ClfImg as ClfImgMNIST
from networks.ConvNetworkImgClfSVHN import ClfImgSVHN
from networks.ConvNetworkTextClf import ClfText as ClfText

from flags.flags_svhnmnist import parser

from datasets.SVHNMNISTDataset import SVHNMNIST
from utils.transforms import get_transform_mnist
from utils.transforms import get_transform_svhn
from utils.filehandling import create_dir_structure
from utils import utils


# global variables
SEED = None 
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)


def get_10_mnist_samples(flags, svhnmnist, num_testing_images):
    samples = []
    for i in range(10):
        while True:
            img_mnist, img_svhn, text, target = svhnmnist.__getitem__(random.randint(0, num_testing_images-1))
            if target == i:
                img_mnist = img_mnist.to(flags.device)
                img_svhn = img_svhn.to(flags.device)
                text = text.to(flags.device);
                samples.append((img_mnist, img_svhn, text, target))
                break;
    return samples


def training_svhnmnist(FLAGS):
    global SEED

    # load data set and create data loader instance
    print('Loading MNIST-SVHN-Text dataset...')
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    FLAGS.num_features = len(alphabet)

    transform_mnist = get_transform_mnist(FLAGS);
    transform_svhn = get_transform_svhn(FLAGS);
    transforms = [transform_mnist, transform_svhn];
    svhnmnist_train = SVHNMNIST(FLAGS.dir_data, FLAGS.len_sequence,  alphabet, train=True, transform=transforms,
                                data_multiplications=FLAGS.data_multiplications)
    svhnmnist_test = SVHNMNIST(FLAGS.dir_data, FLAGS.len_sequence,  alphabet, train=False, transform=transforms,
                               data_multiplications=FLAGS.data_multiplications)

    use_cuda = torch.cuda.is_available();
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu');
    # load global samples
    test_samples = get_10_mnist_samples(FLAGS, svhnmnist_test, num_testing_images=svhnmnist_test.__len__())

    # model definition
    vae_trimodal = VAEtrimodalSVHNMNIST(FLAGS);

    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        vae_trimodal.load_state_dict(torch.load(os.path.join(FLAGS.dir_checkpoints, FLAGS.vae_trimodal_save)));
    FLAGS.rec_weight_m1 = vae_trimodal.rec_w1;
    FLAGS.rec_weight_m2 = vae_trimodal.rec_w2;
    FLAGS.rec_weight_m3 = vae_trimodal.rec_w3;

    model_clf_svhn = None;
    model_clf_mnist = None;
    model_clf_text = None;
    if FLAGS.use_clf:
        model_clf_mnist = ClfImgMNIST();
        model_clf_mnist.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m1)))
        model_clf_svhn = ClfImgSVHN();
        model_clf_svhn.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m2)))
        model_clf_text = ClfText(FLAGS);
        model_clf_text.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m3)))

    vae_trimodal = vae_trimodal.to(FLAGS.device);
    if model_clf_text is not None:
        model_clf_text = model_clf_text.to(FLAGS.device);
    if model_clf_mnist is not None:
        model_clf_mnist = model_clf_mnist.to(FLAGS.device);
    if model_clf_svhn is not None:
        model_clf_svhn = model_clf_svhn.to(FLAGS.device);

    # optimizer definition
    auto_encoder_optimizer = optim.Adam(
        list(vae_trimodal.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2))

    # initialize summary writer
    writer = SummaryWriter(FLAGS.dir_logs)

    str_flags = utils.save_and_log_flags(FLAGS);
    writer.add_text('FLAGS', str_flags, 0)

    print('training epochs progress:')
    it_num_batches = 0;
    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        utils.printProgressBar(epoch, FLAGS.end_epoch)
        # one epoch of training and testing
        it_num_batches, clf_lr = run_epoch(epoch, vae_trimodal,
                                           auto_encoder_optimizer,
                                           svhnmnist_train,
                                           writer, alphabet,
                                           train=True, flags=FLAGS,
                                           model_clf_svhn=model_clf_svhn,
                                           model_clf_mnist=model_clf_mnist,
                                           model_clf_text=model_clf_text,
                                           clf_lr=None,
                                           step_logs=it_num_batches)

        with torch.no_grad():
            it_num_batches, clf_lr = run_epoch(epoch, vae_trimodal,
                                               auto_encoder_optimizer,
                                               svhnmnist_test,
                                               writer, alphabet,
                                               test_samples,
                                               train=False, flags=FLAGS,
                                               model_clf_svhn=model_clf_svhn,
                                               model_clf_mnist=model_clf_mnist,
                                               model_clf_text=model_clf_text,
                                               clf_lr=clf_lr,
                                               step_logs=it_num_batches)

        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
            dir_network_epoch = os.path.join(FLAGS.dir_checkpoints, str(epoch).zfill(4));
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch);
            vae_trimodal.save_networks()
            torch.save(vae_trimodal.state_dict(), os.path.join(dir_network_epoch, FLAGS.vae_trimodal_save))


if __name__ == '__main__':
    FLAGS = parser.parse_args()

    if FLAGS.method == 'poe':
        FLAGS.modality_poe=True;
        FLAGS.poe_unimodal_elbos=True;
    elif FLAGS.method == 'moe':
        FLAGS.modality_moe=True;
    elif FLAGS.method == 'jsd':
        FLAGS.modality_jsd=True;
    else:
        print('method not implemented...exit!')
        sys.exit();

    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content];
    create_dir_structure(FLAGS)
    training_svhnmnist(FLAGS);
