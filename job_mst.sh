

DEBUG=false
LOGDIR=""
METHOD="jsd"
LIKELIHOOD_M1="laplace"
LIKELIHOOD_M2="laplace"
LIKELIHOOD_M3="categorical"
# BASE_DIR needs to be set by the user
BASE_DIR=""
DIR_DATA="${BASE_DIR}/data"
DIR_CLF="${BASE_DIR}/trained_classifiers/trained_clfs_mst"
DIR_EXPERIMENT_BASE="${BASE_DIR}/experiments/mmjsd"
DIR_EXPERIMENT="${DIR_EXPERIMENT_BASE}/MNIST_SVHN_Text/${METHOD}/non_factorized/${LIKELIHOOD_M1}_${LIKELIHOOD_M2}_${LIKELIHOOD_M3}"
PATH_INC_V3="${BASE_DIR}/pt_inception-2015-12-05-6726825d.pth"
DIR_FID="/tmp/MNIST_SVHN_text"

source activate vae

python main_svhnmnist.py --dir_data=$DIR_DATA \
			 --dir_clf=$DIR_CLF \
			 --dir_experiment=$DIR_EXPERIMENT \
			 --inception_state_dict=$PATH_INC_V3 \
			 --dir_fid=$DIR_FID \
			 --method=$METHOD \
			 --poe_unimodal_elbos=True \
			 --style_mnist_dim=0 \
			 --style_svhn_dim=0 \
			 --style_text_dim=0 \
			 --class_dim=20 \
			 --beta=2.5 \
			 --likelihood_m1=$LIKELIHOOD_M1 \
			 --likelihood_m2=$LIKELIHOOD_M2 \
			 --likelihood_m3=$LIKELIHOOD_M3 \
             --div_weight_m1_content=0.25 \
             --div_weight_m2_content=0.25 \
             --div_weight_m3_content=0.25 \
             --div_weight_uniform_content=0.25 \
			 --batch_size=256 \
			 --initial_learning_rate=0.0005 \
			 --eval_freq=20 \
			 --eval_freq_prd=100 \
			 --data_multiplications=20 \
			 --num_hidden_layers=1 \
			 --end_epoch=100 \
			 --calc_nll \
			 --eval_lr \
			 --use_clf \
			 --calc_prd \

