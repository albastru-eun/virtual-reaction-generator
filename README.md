# **Virtual Reaction Generator (VRG)**

<p align="left">
 <img src = "readme/example_route.jpg">
</p>

<br>

***

> * **installation** <br>
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

<br>

***

<br>

### **1. Atom Changer**

main_atom_changer.ipynb <br>
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

main_atom_remover.ipynb <br>
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

main_functionalizer.ipynb <br>
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

main_graph_mixer.ipynb <br>
dataset\[1\]: number of subsituents around the changed ring # remove it before use <br>
id: (id1, id2) # Partial ring from id1 was used to replace part of id2.

```python
#graph mixer (example)
nrows = 50000
similarity_value = 0.8
n_iter=10
```

**output**
<p align="left">
 <img src = "readme/gm_real.png" style="width:55%; height:55%;">
</p>

***

<br>

### **5. Linear Model**

main_linear.ipynb <br>
*or* <br>
main_linear.py # for Linux

<br>

dataset\[1\]: mixed # remove it before use

### * *example of utilization* <br>
(example) 500 rows from train > module 4 (10 iter) > module 3 (5 iter) > module 2 (5 iter) > + 725 rows <br>
<p align="left">
 <img src = "readme/linear_real.jpg">
</p>

<br>
<br>

### **benchmark model**

r-smiles: github.com/otori-bird/retrosynthesis
