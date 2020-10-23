import sys, os
import resource
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from networks.VAEbimodalCelebA import VAEbimodalCelebA
from networks.ConvNetworkImgClfCelebA import ClfImg
from networks.ConvNetworkTextClfCeleba import ClfText

from datasets.CelebADataset import CelebaDataset

from training.training_celeba import run_epoch

from utils import utils
from utils.transforms import get_transform_celeba
from utils.filehandling import create_dir_structure

from flags.flags_celeba import parser

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# global variables
NUM_ATTRIBUTES = 42;
SEED = None
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)


def get_10_celeba_samples(flags, dataset):
    samples = []
    for i in range(10):
        ix = np.random.randint(0, len(dataset.img_names))
        img, text, target = dataset.__getitem__(ix)
        img = img.to(flags.device);
        text = text.to(flags.device);
        samples.append((img, text, target))
    return samples


def training_celeba(FLAGS):
    global SEED

    # load data set and create data loader instance
    print('Loading CelebA (multimodal) dataset...')
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    FLAGS.num_features = len(alphabet)

    transform = get_transform_celeba(FLAGS);
    train_dataset = CelebaDataset(FLAGS, alphabet, partition=0, transform=transform)
    eval_dataset = CelebaDataset(FLAGS, alphabet, partition=1, transform=transform)

    FLAGS.num_samples_train = train_dataset.__len__()
    FLAGS.num_samples_eval = eval_dataset.__len__()

    use_cuda = torch.cuda.is_available();
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu');

    # load global samples
    test_samples = get_10_celeba_samples(FLAGS, eval_dataset)

    # model definition
    vae_bimodal = VAEbimodalCelebA(FLAGS)
    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        vae_bimodal.load_state_dict(torch.load(os.path.join(FLAGS.dir_checkpoints, FLAGS.encoder_save_m1)))

    model_clf_img = None;
    model_clf_text = None;
    print('classifier: ' + str(FLAGS.use_clf))
    if FLAGS.use_clf:
        model_clf_img = ClfImg(flags=FLAGS);
        model_clf_img.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m1)))
        model_clf_text = ClfText(flags=FLAGS);
        model_clf_text.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf, FLAGS.clf_save_m2)))

    vae_bimodal = vae_bimodal.to(FLAGS.device);
    model_clf_img = model_clf_img.to(FLAGS.device);
    model_clf_text = model_clf_text.to(FLAGS.device);

    # optimizer definition
    auto_encoder_optimizer = optim.Adam(
        list(vae_bimodal.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2))

    # initialize summary writer
    writer = SummaryWriter(FLAGS.dir_logs)
    str_flags = utils.save_and_log_flags(FLAGS);
    writer.add_text('FLAGS', str_flags, 0)

    it_num_batches = 0;
    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('Epoch #' + str(epoch))

        # one epoch of training and testing
        it_num_batches, clf_lr, writer = run_epoch(epoch, vae_bimodal,
                                                   auto_encoder_optimizer,
                                                   train_dataset,
                                                   writer, alphabet,
                                                   train=True, flags=FLAGS,
                                                   model_clf_img=model_clf_img,
                                                   model_clf_text=model_clf_text,
                                                   clf_lr=None,
                                                   step_logs=it_num_batches)

        with torch.no_grad():
            it_num_batches, clf_lr, writer = run_epoch(epoch, vae_bimodal,
                                                       auto_encoder_optimizer,
                                                       eval_dataset,
                                                       writer, alphabet,
                                                       test_samples,
                                                       train=False, flags=FLAGS,
                                                       model_clf_img=model_clf_img,
                                                       model_clf_text=model_clf_text,
                                                       clf_lr=clf_lr,
                                                       step_logs=it_num_batches)

        # save checkpoints after every 50 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
            torch.save(vae_bimodal.state_dict(), os.path.join(FLAGS.dir_checkpoints, FLAGS.vae_bimodal_save))
            vae_bimodal.save_networks();


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
        print('method implemented...exit!')
        sys.exit();
    
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content, FLAGS.div_weight_m2_content];
    create_dir_structure(FLAGS, train=(not FLAGS.load_saved))
    training_celeba(FLAGS)
