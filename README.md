# mmJSD

This is the official code repository for the paper "Multimodal Generative
Learning Utilizing Jensen-Shannon-Divergence" which is accepted at NeurIPS
2020.([paper link](https://arxiv.org/abs/2006.08242))

Still work in progress... in case of questions/problems, do not hesitate to reach out to us!

## Preliminaries

This code was developed and tested with:
- Python version 3.5.6
- PyTorch version 1.4.0
- CUDA version 11.0
- The conda environment defined in `environment.yml`

First, set up the conda enviroment as follows:
```bash
conda env create -f environment.yml  # create conda env
conda activate mmjsd                 # activate conda env
```

Second, download the data, inception network, and pretrained classifiers:
```bash
curl -L -o tmp.zip https://www.dropbox.com/sh/lx8669lyok9ois6/AADmH2Q6T_iIlRg2Hp-R_Clca?dl=0
unzip tmp.zip
unzip celeba_data.zip -d data/
unzip data_mnistsvhntext.zip -d data/
```

## Experiments

Experiments can be started by running the respective `job_*` script.
To choose between running the MVAE, MMVAE, and MoPoE-VAE, one needs to
change the script's `METHOD` variabe to "poe", "moe", or "jsd"
respectively.  By default, each experiment uses `METHOD="jsd"`.
Before running any training jobs, please make sure that you have set the paths correctly.

### running MNIST-SVHN-Text
```bash
./job_mst
```

### running Bimodal Celeba
```bash
./job_celeba
```
