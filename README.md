# **Virtual Reaction Generator (VRG)**

<p align="left">
 <img src = "readme/example_route.jpg">
</p>

<br>

***

> * **installation**
*VRG was developed in jupyter-lab under anaconda environment.*

```anaconda
conda create –n vrg python==3.12
conda activate vrg
conda install numpy pandas tqdm scikit-learn scipy
conda install -c conda-forge rdkit
conda install -c conda-forge jupyter-lab
jupyter-lab
```

<br>

> * **Module 1.** [`Atom Changer`](#1-atom-changer)
> * **Module 2.** [`Atom Remover`](#2-atom-remover)
> * **Module 3.** [`Functionalizer`](#3-functionalizer)
> * **Module 4.** [`Graph Mixer`](#4-graph-mixer)
> * **Linear Augmentation** [`Linear Model`](#5-linear-model)
> * *no need to use each model individually, as they can instead be combined using a linear model*

<br>

### [trained model and augmented datasets (pt file, click here)](https://drive.google.com/drive/folders/172dqjaaZn5Gm1YJr_R5Xm1TgrLuBQ0SA)
##### &nbsp;&nbsp; [datasets are also available on Zenodo](https://doi.org/10.5281/zenodo.20319492)
&nbsp;&nbsp; *Kim, H. E., Chung, W.-. jin ., & Kim, H. W. (2026). Augmented USPTO-50k Datasets by Virtual Reaction Generator (VRG) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20319492*

<br>

### VRG_models
This repository contains datasets, trained models, and test resources used for the VRG-based reaction augmentation framework.

---
### Repository Structure

```text
VRG_models/
├── datasets/
│   ├── ESI_Figure_S1/
│   ├── linear_augmentation/
│   ├── original_datasets/
│   └── single_augmentation/
│
├── test_dataset_translation/
│
└── trained_models/
```
---

### Folder Description

#### `datasets/`

Contains datasets generated and used throughout this work.

##### `ESI_Figure_S1/`

Datasets used for the experiments reported in **ESI Figure S1**.

##### `linear_augmentation/`

Datasets generated using the **linear augmentation strategy**.

##### `original_datasets/`

Original datasets.

##### `single_augmentation/`

Datasets generated using **single-module augmentation**.

---

#### `test_dataset_translation/`
Contains translated or processed test datasets used for model evaluation and benchmarking.
All results were given as ~_detailed_results.csv, ~_final_results.txt, and ~_translated.txt

---

#### `trained_models/`
Contains pretrained and trained VRG-based reaction prediction models.

<br>

---

### Notes

* All datasets are provided in reaction SMILES format unless otherwise specified.
* Validation and test datasets were kept unchanged during augmentation experiments.
* Additional experimental details can be found in the Supporting Information (ESI).

<br>

***

<br>

### **1. Atom Changer**

main_atom_changer.ipynb
dataset\[1\]: reaction center # remove it before use

```python
#atom changer (example)
nrows = 50000
similarity_value = 0.85
n_iter=5
```

**output**
<p align="left">
 <img src = "readme/ac_real.png" style="width:35%; height:35%;">
</p>

***

<br>

### **2. Atom Remover**

main_atom_remover.ipynb
dataset\[1\]: UNK

```python
#atom remover (example)
nrows = 50000
similarity_value = 0.85
n_iter=15
```

**output**
<p align="left">
 <img src = "readme/ar_real.png" style="width:55%; height:55%;">
</p>

***

<br>

### **3. Functionalizer**

main_functionalizer.ipynb
dataset\[1\]: reaction center # remove it before use

```python
#functionalizer (example)
nrows = 50000
aug_num = 3
aug_num_val = 3
similarity_value = 0.8
daring_value = 0.6
```

**output**
<p align="left">
 <img src = "readme/f_real.png" style="width:40%; height:40%;">
</p>

***

<br>

### **4. Graph Mixer**

main_graph_mixer.ipynb
dataset\[1\]: number of subsituents around the changed ring # remove it before use
id: (id1, id2) # Partial ring from id1 was used to replace part of id2.

```python
#graph mixer (example)
nrows = 50000
similarity_value = 0.8
n_iter=10
aggressive=0 #0 for strain filtering
```

**output**
<p align="left">
 <img src = "readme/gm_real.png" style="width:55%; height:55%;">
</p>

***

<br>

### **5. Linear Model**

main_linear.ipynb
*or*
main_linear.py # for Linux

<br>

dataset\[1\]: mixed # remove it before use

### * *example of utilization* 
(example) 500 rows from train > module 4 (10 iter) > module 3 (5 iter) > module 2 (5 iter) > 725 rows <br>
<p align="left">
 <img src = "readme/linear_real.png" style="width:55%; height:55%;">
</p>

<br>
<br>

### **Benchmark model**

r-smiles: github.com/otori-bird/retrosynthesis
(pretrained with non-augmented datasets)
*see r-smiles_modified.zip for detailed parameters*

All trained models and results were uploaded at https://drive.google.com/drive/folders/172dqjaaZn5Gm1YJr_R5Xm1TgrLuBQ0SA 

<br>

**Environment Preparation**

*The environment setup was identical to that used in the original R-SMILES GitHub repository.*

Please make sure that Anaconda (or Minoconda is installed before proceeding. The appropriate versions of PyTorch and CUDA toolkit may depend on your hardware environment. According to OpenNMT-py requirements, the PyTorch version should not be lower than 1.6.

```anaconda
conda create -n r-smiles python=3.7
conda activate r-smiles
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install pandas==1.3.4
pip install textdistance==4.2.2
conda install rdkit=2020.09.1.0 -c rdkit
pip install OpenNMT-py==2.2.0
```

<br>

Before training, copy the files inside **r-smiles_modified.zip** into the following directory:

```text
r-smiles\pretrain_finetune\finetune\PtoR
```

<br>

Additionally, transfer datasets into the following directory:

```text
r-smiles\dataset\USPTO_50K\raw_train.csv
```

The training procedure was performed in the following order.

<br>

### Step 1. Generate the augmented pretraining dataset

```text
python preprocessing/generate_PtoR_data.py -dataset USPTO_50K -augmentation 20 -processes 8
```

<br>

### Step 2. Train the model using OpenNMT

Run the OpenNMT training command with the prepared configuration file:

```text
onmt_train -config pretrain_finetune/finetune/PtoR/PtoR-50K-aug20-config.yml
```

*If you want to fine-tune the model from a different checkpoint, modify the train_from parameter in the corresponding configuration file.*

<br>

### Step 3. Average checkpoints

Run the prepared shell script to average the checkpoints and obtain the final model checkpoint:

```text
bash pretrain_finetune/finetune/PtoR/PtoR-50K-aug20-average.sh
```

*You may modify the shell script if you want to average different checkpoint ranges.*

<br>

### Step 4. Run inference

Run the OpenNMT translate command with the prepared configuration file:

```text
onmt_translate -config pretrain_finetune/finetune/PtoR/PtoR-50K-aug20-translate.yml
```

*If you want to use a different model checkpoint, modify the model parameter in the corresponding translation file.*

<br>

### Step 5. Evaluate prediction accuracy

```text
python score.py \
-beam_size 10 \
-n_best 10 \
-augmentation 20 \
-targets ./dataset/USPTO_50K_PtoR_aug20/test/tgt-test.txt \
-predictions ./exp/USPTO_50K_PtoR_aug20/average_model_26-30-results.txt \
-process_number 8 \
-score_alpha 1 \
-save_file ./final_results.txt \
-detailed \
-source ./dataset/USPTO_50K_PtoR_aug20/test/src-test.txt
```

<br>
<br>

**retrosynthesis prediction (improved from retrained r-smiles)**

*overall reaction type*

|   modules  | top-1 | top-3 | top-5 | top-10 |
|:----------:|:-----:|:-----:|:-----:|:------:|
| r-original   | 56.3  | 78.7  | 84.7  | 89.7   |
| r-AC         | **+0.2**  | +0.2  | –0.1  | +0.1   |
| r-AR         | +0.0  | **+0.7**  | +0.5  | +0.2   |
| r-FN          | –0.1  | +0.0  | **+0.7**  | **+0.5**   |
| r-GM         | –0.4  | +0.2  | +0.5  | +0.0   |
|   |   |   |   |   |
| r-ensemble   | +0.3    | +1.2    | **+0.9**    | +0.7     |
| r-ensemble'   | **+0.6**    | **+1.5**    | +0.7    | **+0.9**     |

<br>

*overall reaction type (MaxFrag)*

|   modules  | top-1 | top-3 | top-5 | top-10 |
|:----------:|:-----:|:-----:|:-----:|:------:|
| r-original   | 60.7  | 82.3  | 87.6  | 91.7   |
| r-AC         | **+0.3**  | +0.2  | +0.2  | +0.3     |
| r-AR         | +0.0  | **+0.5**  | +0.7  | +0.4     |
| r-F          | –0.1  | +0.0  | **+1.0**  | **+0.7**     |
| r-GM         | –0.3  | +0.3  | +0.4  | +0.1     |
|   |   |   |   |   |
| r-ensemble   | +0.4    | +1.2    | **+1.0**    | +0.5     |
| r-ensemble'   | **+0.7**    | **+1.5**    | +0.9    | +0.6     |

<br>

*acyclic*

|   modules  | top-1 | top-3 | top-5 | top-10 |
|:----------:|:-----:|:-----:|:-----:|:------:|
| r-original   | **58.9**  | 81.4  | 87.3  | 91.9   |
| r-AC         | **+0.0**  | +0.0  | –0.3  | +0.0   |
| r-AR         | –0.2  | **+0.3**  | +0.2  | +0.0   |
| r-FN          | –0.1  | –0.1  | **+0.6**  | **+0.5**   |
| r-GM         | –0.5  | +0.1  | +0.3  | +0.1   |
|   |   |   |   |   |
| r-ensemble   | +0.2    | +1.2    | **+0.7**    | **+0.8**     |
| r-ensemble'   | **+0.6**    | **+1.4**    | +0.6    | **+0.8**     |

<br>

*ring-opening*

|   modules  | top-1 | top-3 | top-5 | top-10 |
|:----------:|:-----:|:-----:|:-----:|:------:|
| r-original   | 30.7  | 52.4  | 60.6  | 67.6   |
| r-AC         | +0.8  | +1.5  | +1.7  | +2.4   |
| r-AR         | +1.4  | **+5.9**  | **+4.5**  | **+3.7**   |
| r-FN          | +0.6  | +0.8  | +1.1  | +3.4   |
| r-GM         | **+2.5**  | +2.5  | +1.7  | +0.6   |
|   |   |   |   |   |
| r-ensemble   | +1.1    | +2.2    | +3.1    | +0.3     |
| r-ensemble'   | +1.4    | +3.1    | +1.7    | +1.7     |

<br>

*ring-closing*

|   modules  | top-1 | top-3 | top-5 | top-10 |
|:----------:|:-----:|:-----:|:-----:|:------:|
| r-original   | 35.5  | 54.5  | 59.5  | **71.9**   |
| r-AC         | **+4.2**  | **+0.9**  | +0.0  | –4.1   |
| r-AR         | +1.7  | **+0.9**  | +0.8  | –1.7   |
| r-FN          | –3.3  | **+0.9**  | +1.7  | –4.8   |
| r-GM         | –4.1  | +0.1  | **+3.3**  | –4.8   |
|   |   |   |   |   |
| r-ensemble   | –0.8    | +0.9   | +1.7    | –1.7     |
| r-ensemble'   | –2.4    | **+1.7**   | +1.7    | –0.8     |

<br>

*with chiral reactant*

|   modules  | top-1 | top-3 | top-5 | top-10 |
|:----------:|:-----:|:-----:|:-----:|:------:|
| r-original   | **52.3**  | 73.3  | 79.7  | 85.9   |
| r-AC         | –0.5  | +0.0  | –0.6  | –0.9   |
| r-AR         | –0.9  | **+0.6**  | +0.1  | –0.3   |
| r-FN          | –1.0  | +0.3  | **+0.7**  | **+0.1**   |
| r-GM         | –0.2  | –0.4  | +0.1  | –0.4   |
|   |   |   |   |   |
| r-ensemble   | –0.7    | +1.9    | **+1.2**    | +0.1     |
| r-ensemble'   | **+0.0**    | **+2.6**    | +0.7    | **+0.7**     |

<br>

*w/o chiral reactant*

|   modules  | top-1 | top-3 | top-5 | top-10 |
|:----------:|:-----:|:-----:|:-----:|:------:|
| r-original   | 57.2  | 79.9  | 85.9  | 90.5   |
| r-AC         | **+0.4**  | +0.3  | +0.0  | +0.4   |
| r-AR         | +0.2  | **+0.8**  | **+0.6**  | +0.4   |
| r-FN          | +0.1  | –0.1  | **+0.6**  | **+0.7**   |
| r-GM         | –0.4  | +0.4  | **+0.6**  | +0.2   |
|   |   |   |   |   |
| r-ensemble   | +0.5    | +1.1    | **+0.8**    | +0.9     |
| r-ensemble'   | **+0.6**    | **+1.3**    | +0.7    | **+1.0**     |

<br>
<br>
