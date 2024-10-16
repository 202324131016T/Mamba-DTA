# Mamba-DTA

> **Paper: Mamba-DTA: Drug-Target Binding Affinity Prediction with State Space Model**
>
> Author: Yulong Wu, Jin Xie, Jing Nie, Xiaohong Zhang and Yuansong Zeng

## Installation

```
conda create -n Mamba-DTA python=3.8
conda activate Mamba-DTA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Other requirements:
- Linux
- NVIDIA GPU
- The detailed environment configuration is shown in **environment.yml**.

## Train and Test

- The **config/arguments.py** provides some model parameters.
- The **datahelper.py** provides some methods for data preprocessing.
- The **emetrics.py** provides evaluation metrics for the model.
- The **figures.py** provides some visualization methods.
- The **model/model_MambaDTA.py** is the main model of our Mamba-DTA, and other **model/model_MambaDTA_\*.py** are the model needed for ablation experiments.
- We use them in **run_experiments.py**.

```
python run_experiments.py
# or
bash run.sh
```

