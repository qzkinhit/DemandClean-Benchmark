## Generalizable Data Cleaning of Tabular Data in Latent Space

<p align="center">
<img src="https://github.com/DataManagementLab/data_cleaning_with_latent_operators/blob/main/PAPERequivariance.png?raw=true" width="420" height="360">

## Citation
Please use the following BibTeX code to cite our [paper](https://www.vldb.org/pvldb/vol17/p4786-reis.pdf), which is accepted for presentation at [VLDB 2024](https://vldb.org/pvldb/volumes/17#issue-13):
```
@inproceedings{lopster2024,  
title       = {Generalizable Data Cleaning of Tabular Data in Latent Space},  
author      = {Eduardo dos Reis, Mohamed Abdelaal and Carsten Binnig},  
journal     = {Proceedings of the VLDB Endowment},  
volume      = {17},  
number      = {13},  
pages       = {4786--4798},  
year        = {2024},  
publisher   = {VLDB Endowment}  
}
```

## Data cleaning on your own dataset
We provided the training and cleaning in a single script **lopster.py**. 

**Step 1**. The requirements are two csv files inside a folder **\< my dataset folder \>**: a  **clean.csv** containing a - mostly - clean sample of the data, and **dirty01.csv**, the target dataset to be cleaned. Keep this folder in the project root folder for ease-of-use.
- \< my dataset folder \>
	+ clean.csv
	+ dirty01.csv
	
**Step 2**
Open the **dataset_configuration.json** file and replace the **"empty"** json entry with your dataset information, while also replacing the name of the entry for **\< my dataset folder \>**.
	
Next, the script will yield a **lopster.csv** file containing the full cleaned version of the dataset under **\< my dataset folder \>** . If it is the first time running the script for the current dataset, it will first train a Lopster model using tensorflow. Example usage:
```
python3 lopster.py --dataset < my dataset folder >
```
You can also change hyperparameters for training (K, batch size, latent space dimensionality, training epochs):
```
python3 lopster.py --dataset < my dataset folder > --path < path/to/model/folder/ > --K 12 --latent 120 --epochs 100 --batch_size 256
```

## Setup for reproducibility
We provided the REIN benchmark  data used for the paper as is, in **rein_data.zip**. Once the data is decompressed to the **DATASETS_REIN/** folder, one must create a python 3 environment and install the dependencies listed on **requirements.txt**. Next, a new Lopster model can be trained on any dataset as long as its configuration is mapped to the **datasets.py** script.

## Reproducing results and Training
To reproduce the paper results on the REIN benchmark, a script was provided (**rein_benchmark.py**). For example:
```
python3 rein_benchmark.py --dataset adult  --K 12 --latent 120 --epochs 100 --batch_size 256 --eval_tuples 30000
```
## Ablation Studies
All our ablation studies are available on **ablation_studies.py**, and it is already configured to replicate only the published ones.
```
python3 ablation_studies.py --dataset soccer_PLAYER  --K 12 --latent 120 --epochs 40 --batch_size 256 --training_tuples 30000
```
## Plots
All the evaluation files are saved inside the **evaluation/** folder as .CSV and can be easily plotted. We provide a script to replicate our plots in **plotter.py**, the usage follows:
```
python3 plotter.py --experiment tuplewise  --dataset adult --y_title  --legend
```
## Datasets
The largest dataset used for the paper experiments is available for download at 
[Large Soccer Dataset](https://drive.google.com/file/d/1svyTShYV6DAO87_BbmWmeUDozf7AAvcv/view?usp=sharing). All the others are available in the DATASETS_REIN folder.