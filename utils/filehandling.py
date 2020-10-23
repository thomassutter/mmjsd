
import os
import shutil
from datetime import datetime

from utils.constants_celeba import CLASSES_STR

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name, ignore_errors=True)
        os.makedirs(dir_name)


def get_str_experiments(flags):
    dateTimeObj = datetime.now()
    dateStr = dateTimeObj.strftime("%Y_%m_%d_%H_%M_%S_%f")
    str_experiments = flags.dataset + '_' + dateStr;
    return str_experiments


def create_dir_structure_testing_celeba(flags):
    for k, label_str in enumerate(CLASSES_STR):
        dir_gen_eval_label = os.path.join(flags.dir_gen_eval, label_str)
        create_dir(dir_gen_eval_label)
        dir_inference_label = os.path.join(flags.dir_inference, label_str)
        create_dir(dir_inference_label)


def create_dir_structure(flags, train=True):
    if train:
        str_experiments = get_str_experiments(flags)
        flags.dir_experiment_run = os.path.join(flags.dir_experiment, str_experiments)
    else:
        flags.dir_experiment_run = flags.dir_experiment;

    print(flags.dir_experiment_run)
    if train:
        create_dir(flags.dir_experiment_run)

    flags.dir_checkpoints = os.path.join(flags.dir_experiment_run, 'checkpoints')
    if train:
        create_dir(flags.dir_checkpoints)

    flags.dir_logs = os.path.join(flags.dir_experiment_run, 'logs')
    if train:
        create_dir(flags.dir_logs)
    print(flags.dir_logs)

    flags.dir_logs_clf = os.path.join(flags.dir_experiment_run, 'logs_clf')
    if train:
        create_dir(flags.dir_logs_clf)

    flags.dir_gen_eval = os.path.join(flags.dir_experiment_run, 'generation_evaluation')
    if train:
        create_dir(flags.dir_gen_eval)

    flags.dir_inference = os.path.join(flags.dir_experiment_run, 'inference')
    if train:
        create_dir(flags.dir_inference)

    if train and flags.dataset == 'CelebA':
        create_dir_structure_testing_celeba(flags);


    if flags.dir_fid is None:
        flags.dir_fid = flags.dir_experiment_run;
    elif not train:
        flags.dir_fid = os.path.join(flags.dir_experiment_run, 'fid_eval');
        if not os.path.exists(flags.dir_fid):
            os.makedirs(flags.dir_fid);
    flags.dir_gen_eval_fid_cond_gen = os.path.join(flags.dir_fid, 'fid', 'conditional_generation')
    flags.dir_gen_eval_fid_real = os.path.join(flags.dir_fid, 'fid', 'real')
    flags.dir_gen_eval_fid_random = os.path.join(flags.dir_fid, 'fid', 'random_sampling')
    flags.dir_gen_eval_fid_dynamicprior = os.path.join(flags.dir_fid, 'fid', 'dynamic_prior')
    create_dir(flags.dir_gen_eval_fid_cond_gen)
    create_dir(flags.dir_gen_eval_fid_real)
    create_dir(flags.dir_gen_eval_fid_random)
    create_dir(flags.dir_gen_eval_fid_dynamicprior)

    flags.dir_gen_eval_fid_cond_gen_1a2m = os.path.join(flags.dir_fid, 'fid', 'cond_gen_1a2m')
    flags.dir_gen_eval_fid_cond_gen_2a1m = os.path.join(flags.dir_fid, 'fid', 'cond_gen_2a1m')
    if flags.dataset == 'SVHN_MNIST_text':
        create_dir(flags.dir_gen_eval_fid_cond_gen_1a2m)
        create_dir(flags.dir_gen_eval_fid_cond_gen_2a1m)


    flags.dir_plots = os.path.join(flags.dir_experiment_run, 'plots')
    if train:
        create_dir(flags.dir_plots)
    flags.dir_swapping = os.path.join(flags.dir_plots, 'swapping')
    if train:
        create_dir(flags.dir_swapping)

    flags.dir_cond_gen = os.path.join(flags.dir_plots, 'cond_gen')
    if train:
        create_dir(flags.dir_cond_gen)

    flags.dir_random_samples = os.path.join(flags.dir_plots, 'random_samples')
    if train:
        create_dir(flags.dir_random_samples)

    flags.dir_cond_gen_1a = os.path.join(flags.dir_plots, 'cond_gen_1a')
    create_dir(flags.dir_cond_gen_1a)
    flags.dir_cond_gen_2a = os.path.join(flags.dir_plots, 'cond_gen_2a')
    if train and flags.dataset == 'SVHN_MNIST_text':
        create_dir(flags.dir_cond_gen_2a)
    return flags;
