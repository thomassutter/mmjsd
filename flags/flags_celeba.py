
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='CelebA',
                    help="name of the dataset")

# add arguments
parser.add_argument('--batch_size', type=int, default=256,
                    help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.00025,
                    help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9,
                    help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999,
                    help="default beta_2 val for adam")

parser.add_argument('--style_m1_dim', type=int, default=32,
                    help="dimension of varying factor latent space")
parser.add_argument('--style_m2_dim', type=int, default=32,
                    help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=32,
                    help="dimension of common factor latent space")

#data
parser.add_argument('--dir_data', type=str, default='../data',
                    help="directory where data is stored")
parser.add_argument('--dir_text', type=str, default='../text',
                    help="directory where text is stored")
parser.add_argument('--len_sequence', type=int, default=256,
                    help="length of sequence")
parser.add_argument('--random_text_ordering', type=bool, default=False,
                    help="flag to indicate if attributes are shuffled randomly")
parser.add_argument('--random_text_startindex', type=bool, default=True,
                    help="flag to indicate if start index is random")
parser.add_argument('--img_size', type=int, default=64,
                    help="img dimension (width/height)")
parser.add_argument('--image_channels', type=int, default=3,
                    help="number of channels in images")
parser.add_argument('--crop_size_img', type=int, default=148,
                    help="number of channels in images")

parser.add_argument('--DIM_text', type=int, default=128,
                    help="filter dimensions of residual layers")
parser.add_argument('--DIM_img', type=int, default=128,
                    help="filter dimensions of residual layers")
parser.add_argument('--num_layers_text', type=int, default=7,
                    help="number of residual layers")
parser.add_argument('--num_layers_img', type=int, default=5,
                    help="number of residual layers")
parser.add_argument('--kernelsize_enc_text', type=int, default=3,
                    help="kernel size encoder")
parser.add_argument('--kernelsize_dec_text', type=int, default=3,
                    help="kernel size decoder")
parser.add_argument('--kernelsize_enc_img', type=int, default=3,
                    help="kernel size encoder")
parser.add_argument('--kernelsize_dec_img', type=int, default=3,
                    help="kernel size decoder")
parser.add_argument('--enc_stride_text', type=int, default=2,
                    help="stride encoder")
parser.add_argument('--dec_stride_text', type=int, default=2,
                    help="stride decoder")
parser.add_argument('--enc_stride_img', type=int, default=2,
                    help="stride encoder")
parser.add_argument('--dec_stride_img', type=int, default=2,
                    help="stride decoder")
parser.add_argument('--enc_padding_text', type=int, default=1,
                    help="padding encoder")
parser.add_argument('--dec_padding_text', type=int, default=1,
                    help="padding decoder")
parser.add_argument('--dec_outputpadding_text', type=int, default=1,
                    help="output padding decoder")
parser.add_argument('--enc_padding_img', type=int, default=1,
                    help="padding encoder")
parser.add_argument('--dec_padding_img', type=int, default=1,
                    help="padding decoder")
parser.add_argument('--dec_outputpadding_img', type=int, default=1,
                    help="output padding decoder")
parser.add_argument('--enc_dilation_text', type=int, default=1,
                    help="dilation encoder")
parser.add_argument('--dec_dilation_text', type=int, default=1,
                    help="dilation decoder")
parser.add_argument('--compression_power', type=int, default=2,
                    help="compression power")
parser.add_argument('--a_text', type=float, default=2.0,
                    help="residual weight in residual layers")
parser.add_argument('--b_text', type=float, default=0.3,
                    help="convolution weight in residual layers")
parser.add_argument('--a_img', type=float, default=2.0,
                    help="residual weight in residual layers")
parser.add_argument('--b_img', type=float, default=0.3,
                    help="convolution weight in residual layers")
parser.add_argument('--likelihood_m1', type=str, default='laplace',
                    help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='categorical',
                    help="output distribution")


#classifier
parser.add_argument('--dir_clf', type=str, default='../clf',
                    help="directory where clf is stored")
parser.add_argument('--clf_save_m1', type=str, default='clf_m1',
                    help="model save for clf")
parser.add_argument('--clf_save_m2', type=str, default='clf_m2',
                    help="model save for clf")

parser.add_argument('--eval_freq', type=int, default=5,
                    help=("frequency of evaluation of latent representation of"
                          "generative performance (in number of epochs)"))
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


# paths to save models
parser.add_argument('--encoder_save_m1', type=str, default='encoderM1',
                    help="model save for encoder")
parser.add_argument('--encoder_save_m2', type=str, default='encoderM2',
                    help="model save for encoder")
parser.add_argument('--decoder_save_m1', type=str, default='decoderM1',
                    help="model save for decoder")
parser.add_argument('--decoder_save_m2', type=str, default='decoderM2',
                    help="model save for decoder")
parser.add_argument('--vae_bimodal_save', type=str, default='vae_bimodal',
                    help="model save for vae_bimodal")

parser.add_argument('--load_saved', type=bool, default=False,
                    help="flag to indicate if a saved model will be loaded")
parser.add_argument('--start_epoch', type=int, default=0,
                    help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100,
                    help="flag to indicate the final epoch of training")

#file structure
parser.add_argument('--dir_experiment', type=str, default='../experiments',
                    help="directory to save generated samples in")
parser.add_argument('--dir_fid', type=str, default=None,
                    help="directory to save generated samples for fid score calculation")
parser.add_argument('--save_plot_images', action='store_true', default=False,
                    help="save plots additionally to disk")

#multimodal
parser.add_argument('--method', type=str, default='poe',
                    help='choose method for training the model')
parser.add_argument('--bimodal', type=bool, default=True,
                    help="flag to indicate if a bimodal model should be run")
parser.add_argument('--modality_jsd', type=bool, default=False,
                    help="modality_jsd")
parser.add_argument('--modality_poe', type=bool, default=False,
                    help="modality_poe")
parser.add_argument('--modality_moe', type=bool, default=False,
                    help="modality_moe")
parser.add_argument('--unimodal_klds', type=bool, default=False,
                    help="unimodal_klds")
parser.add_argument('--mixture_prior', type=bool, default=False,
                    help="mixture prior")
parser.add_argument('--broadcast_decoder', type=bool, default=False,
                    help="broadcast_decoder")
parser.add_argument('--factorized_representation', action='store_true', default=False,
                    help="factorized_representation")

#weighting of loss terms
parser.add_argument('--beta', type=float, default=2.5,
                    help="default weight of sum of weighted divergence terms")
parser.add_argument('--beta_style', type=float, default=2.0,
                    help=("default weight of sum of weighted style"
                          " divergence terms"))
parser.add_argument('--beta_content', type=float, default=1.0,
                    help=("default weight of sum of weighted content"
                          " divergence terms"))
parser.add_argument('--beta_m1_style', type=float, default=1.0,
                    help="default weight divergence term style modality 1")
parser.add_argument('--beta_m2_style', type=float, default=2.0,
                    help="default weight divergence term style modality 2")
parser.add_argument('--div_weight_m1_content', type=float, default=0.35,
                    help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=0.35,
                    help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_uniform_content', type=float, default=0.3,
                    help="default weight divergence term prior")
parser.add_argument('--rec_weight_m1', type=float, default=0.5,
                    help="weighting of reconstruction vs. divergence")
parser.add_argument('--rec_weight_m2', type=float, default=0.5,
                    help="weighting of reconstruction vs. divergence")




