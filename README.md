# DLGN (De Novo Generation of Dual-Ligand Molecules Using Adversarial Training and Reinforcement Learning)
This is an PyTorch implementation of DLGN.

# Requirement
Modern NVIDIA GPU  </br>
CUDA 10.1 </br>
python 3.7 </br>
pytorch 1.5.1  </br>
RDKit  2020.03.2.0  </br>
scikit-learn 0.24.0  </br>
Numpy 1.18.5  </br>
Fcd   1.1 (see https://github.com/bioinf-jku/FCD )  </br>
MOSES （see https://github.com/molecularsets/moses) </br>
tqdm  4.46.1 </br>


# Step1: Pre-training a prior generator
In the training process, ChEMBL dataset is used to pretrain a prior generator to learn basic SMILES grammar. Here we use Teacher’s forcing  which uses the preceding ground truth tokens in the string instead of the tokens previously predicted by the network as input and maximizes the predicted probability of the next ground truth tokens at each step. It is actually a Maximum Likelihood Estimate. To train the model, run

```
python MLE.py 
```

# Step2: Fire-tuning by using DLGN framework
The DLGN generator is initialized as the pretrained prior network, and then fine-tuned by training on DRD2/HTR1A training datasets via adversarial training and policy gradient. 
To train the model, run

```
python PolicyGradient.py 
```

# Step3: Evaluation
```
python test.py
```
