
DEBUG=false
LOGDIR=""
METHOD="jsd"
DATASET="CelebA"
LIKELIHOOD_M1="laplace"
LIKELIHOOD_M2="categorical"
BASE_DIR="/usr/scratch/projects/multimodality"
DIR_DATA="${BASE_DIR}/data"
DIR_CLF="${BASE_DIR}/experiments/trained_classifiers/${DATASET}"
DIR_TEXT="${DIR_DATA}/${DATASET}"
DIR_EXPERIMENT_BASE="${BASE_DIR}/experiments/mmjsd"
DIR_EXPERIMENT="${DIR_EXPERIMENT_BASE}/${DATASET}/${METHOD}/factorized/${LIKELIHOOD_M1}_${LIKELIHOOD_M2}"
PATH_INC_V3="${BASE_DIR}/experiments/inception_v3/pt_inception-2015-12-05-6726825d.pth"
DIR_FID="/tmp/CelebA"

source activate vae


python main_celeba.py --dir_data=$DIR_DATA \
	              --dir_text=$DIR_TEXT \
	              --dir_clf=$DIR_CLF \
	              --dir_experiment=$DIR_EXPERIMENT \
	              --inception_state_dict=$PATH_INC_V3 \
	              --dir_fid=$DIR_FID \
		      --method=$METHOD \
	              --beta=2.5 \
	              --beta_style=2.0 \
	              --beta_content=1.0 \
	              --beta_m1_style=1.0 \
	              --beta_m2_style=5.0 \
	              --div_weight_m1_content=0.35 \
	              --div_weight_m2_content=0.35 \
	              --div_weight_uniform_content=0.3 \
	              --likelihood_m1=$LIKELIHOOD_M1 \
	              --likelihood_m2=$LIKELIHOOD_M2 \
	              --batch_size=256 \
	              --initial_learning_rate=0.0005 \
	              --eval_freq=1 \
	              --end_epoch=250 \
	              --factorized_representation \
	              --calc_nll \
	              --eval_lr \
	              --use_clf \
	              --calc_prd \
	


