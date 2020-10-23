
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='SVHN_MNIST_text',
                    help="name of the dataset")

# training
parser.add_argument('--batch_size', type=int, default=256,
                    help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.001,
                    help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9,
                    help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999,
                    help="default beta_2 val for adam")
parser.add_argument('--start_epoch', type=int, default=0,
                    help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=50,
                    help="flag to indicate the final epoch of training")


parser.add_argument('--style_mnist_dim', type=int, default=4,
                    help="dimension of varying factor latent space")
parser.add_argument('--style_svhn_dim', type=int, default=4,
                    help="dimension of varying factor latent space")
parser.add_argument('--style_text_dim', type=int, default=4,
                    help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=16,
                    help="dimension of common factor latent space")
parser.add_argument('--num_classes', type=int, default=10,
                    help="number of classes on which the data set trained")

parser.add_argument('--len_sequence', type=int, default=8,
                    help="length of sequence")
parser.add_argument('--img_size_mnist', type=int, default=28,
                    help="img dimension (width/height)")
parser.add_argument('--num_channels_mnist', type=int, default=1,
                    help="number of channels in images")
parser.add_argument('--img_size_svhn', type=int, default=32,
                    help="img dimension (width/height)")
parser.add_argument('--num_channels_svhn', type=int, default=3,
                    help="number of channels in images")
parser.add_argument('--dim', type=int, default=64,
                    help="number of classes on which the data set trained")
parser.add_argument('--data_multiplications', type=int, default=20,
                    help="number of pairs per sample")
parser.add_argument('--likelihood_m1', type=str, default='laplace',
                    help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='laplace',
                    help="output distribution")
parser.add_argument('--likelihood_m3', type=str, default='categorical',
                    help="output distribution")
parser.add_argument('--num_hidden_layers', type=int, default=1,
                    help="number of channels in images")

# paths to save models
parser.add_argument('--encoder_save_m1', type=str, default='encoderM1',
                    help="model save for encoder")
parser.add_argument('--encoder_save_m2', type=str, default='encoderM2',
                    help="model save for encoder")
parser.add_argument('--encoder_save_m3', type=str, default='encoderM3',
                    help="model save for decoder")
parser.add_argument('--decoder_save_m1', type=str, default='decoderM1',
                    help="model save for decoder")
parser.add_argument('--decoder_save_m2', type=str, default='decoderM2',
                    help="model save for decoder")
parser.add_argument('--decoder_save_m3', type=str, default='decoderM3',
                    help="model save for decoder")
parser.add_argument('--vae_trimodal_save', type=str, default='vae_trimodal',
                    help="model save for vae_bimodal")
parser.add_argument('--load_saved', type=bool, default=False,
                    help="flag to indicate if a saved model will be loaded")

#classifiers
parser.add_argument('--dir_clf', type=str, default='../clf',
                    help="directory where clf is stored")
parser.add_argument('--clf_save_m1', type=str, default='clf_m1',
                    help="model save for clf")
parser.add_argument('--clf_save_m2', type=str, default='clf_m2',
                    help="model save for clf")
parser.add_argument('--clf_save_m3', type=str, default='clf_m3',
                    help="model save for clf")

parser.add_argument('--eval_freq', type=int, default=10,
                    help=("frequency of evaluation of latent representation" \
                           "of generative performance (in number of epochs)"))
parser.add_argument('--eval_freq_prd', type=int, default=100,
                    help=("frequency of evaluation of latent representation of" \
                          "generative performance (in number of epochs)"))
parser.add_argument('--num_samples_fid', type=int, default=10000,
                    help="number of samples for fid calculation")
parser.add_argument('--calc_nll', action='store_true', default=False,
                    help="flag to indicate calculation of nll")
parser.add_argument('--use_clf', action='store_true', default=False,
                    help="flag to indicate if generates samples should be classified")
parser.add_argument('--eval_lr', action='store_true', default=False,
                    help="flag to indicate if represenations will be evaluated")
parser.add_argument('--calc_prd', action='store_true', default=False,
                    help=("flag to indicate if generated samples should be"
                          "evaluated on their generative quality"))

#fid_score
parser.add_argument('--inception_state_dict', type=str,
                    default='../inception_state_dict.pth',
                    help="path to inception v3 state dict")

#data
parser.add_argument('--dir_data', type=str, default='./data',
                    help="directory where data is stored")

#file structure
parser.add_argument('--dir_experiment', type=str, default='./experiments',
                    help="directory to save experiments output")
parser.add_argument('--dir_fid', type=str, default='./fid',
                    help="directory to save generated samples in")
parser.add_argument('--save_plot_images', action='store_true', default=False,
                    help="save plots additionally to disk")

#multimodal
parser.add_argument('--method', type=str, default='poe',
                    help='choose method for training the model')
parser.add_argument('--modality_jsd', type=bool, default=False,
                    help="modality_jsd")
parser.add_argument('--modality_poe', type=bool, default=False,
                    help="modality_poe")
parser.add_argument('--modality_moe', type=bool, default=False,
                    help="modality_moe")
parser.add_argument('--poe_unimodal_elbos', type=bool, default=False,
                    help="unimodal_klds")
parser.add_argument('--subsampled_reconstruction',  default=True,
                    help="subsample reconstruction path")
parser.add_argument('--factorized_representation', action='store_true',
                    default=False,
                    help="factorized_representation")
parser.add_argument('--include_prior_expert', action='store_true',
                    default=False,
                    help="factorized_representation")

#weighting of loss terms
parser.add_argument('--beta', type=float, default=5.0,
                    help="default weight of sum of weighted divergence terms")
parser.add_argument('--beta_style', type=float, default=3.0,
                    help="default weight of sum of weighted style divergence terms")
parser.add_argument('--beta_content', type=float, default=1.0,
                    help="default weight of sum of weighted content divergence terms")
parser.add_argument('--beta_m1_style', type=float, default=1.0,
                    help="default weight divergence term style modality 1")
parser.add_argument('--beta_m2_style', type=float, default=1.0,
                    help="default weight divergence term style modality 2")
parser.add_argument('--beta_m3_style', type=float, default=5.0,
                    help="default weight divergence term style modality 2")
parser.add_argument('--div_weight_m1_content', type=float, default=0.25,
                    help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=0.25,
                    help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_m3_content', type=float, default=0.25,
                    help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_uniform_content', type=float, default=0.25,
                    help="default weight divergence term prior")
parser.add_argument('--adaptive_rec_weights', action='store_true',
                    default=False, help="use adaptive rec weights")


#annealing
parser.add_argument('--kl_annealing', type=int, default=0,
                    help=("number of kl annealing steps;"
                          " 0 if no annealing should be done"))
