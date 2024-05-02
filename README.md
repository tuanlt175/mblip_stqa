#  Joint Training and Feature Augmentation for Vietnamese Visual Reading Comprehension

## Contents
1. [Install](#setup) <br>
2. [Train model](#train_model) <br>
3. [Evaluate model](#evaluate_model) <br>

Our model is available at [letuan/mblip-mt0-xl-vivqa ](https://huggingface.co/letuan/mblip-mt0-xl-vivqa)

## 1. Install <a name="setup"></a>
Env requirements:
```
python 3.8
cuda 11.7
```

Installation Scripts:
```bash
apt-get install libcusparse-dev-11-2 libcublas-dev-11-2 libcusolver-dev-11-2 libcurand-dev-11-2
conda create -n vivrc python=3.8 --yes
source activate vivrc \
    && pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu116\
    && pip install -r requirements.txt \
    && pip install -e . \
    && pip install --user ipykernel \
    && pip install ipywidgets \
    && python -m ipykernel install --user --name=vivrc

```

## 2. Train model <a name="train_model"></a>
```bash
chmod +x deepspeed_train.sh
./deepspeed_train.sh
```

## 3. Evaluate model <a name="evaluate_model"></a>
```bash
chmod +x evaluate.sh
./evaluate.sh
```


