# CWIDTI

## Overview 

This Python script is used to train, validate, test deep learning model for prediction of drug-target interaction (DTI) Deep learning model will be built by Keras with tensorflow. You can set almost hyper-parameters as you want, See below parameter description DTI, drug, target protein and their interaction data must be written as txt file format. And feature should be tab-delimited format for script to parse data. 

## Requirement

```
tensorflow >  2.0
keras > 2.0 
numpy
pandas 
matplotlib
scikit-learn  
```

## Usage 

```
usage: python Preprocessing_D.py DrugBank
       python Preprocessing_T.py DrugBank
       python Global processing_D.py 
       python Global processing_T.py 
       python CWI-DTI.py CWI_params.txt
       python Predicted__S1.py
```

## Data Specification

All training, validation, test should follow specification to be parsed correctly by CWI-DTI

  * Model takes 3 types data as a set, Drug-target interaction data, target protein data, compound data.

  * They should be `.txt` format.

  * For feature column, each dimension of features in columns should be delimited with tab (`\t`)

After three data are correctly listed, target protein data and compound data will be joined with drug-target data, generating DTI feature.

### Drug target interaction data

Drug target interaction data should be at least 2 columns `Drug_ID` and `Target_ID`,

and should have `Label` column except `--test` case. `Label` colmun has to have label `0` as negative and `1` as positive.

| Drug_ID | Target_ID | Label |
| ------- | --------- | ----- |
| CWI0001 | T0001     | 0     |
| ...     | ...       | ...   |
| CWI0100 | T0100     | 1     |

### Chinese and Western medicine data

`Drug_ID` column will be used as forein key from `Drug_ID` from Drug-target interaction data.

| Western medicine_ID | Smile                |
| :------------------ | :------------------- |
| WM0001              | N1N=CC=CC2=CC=CC=C12 |

| Chinese medicine_ID | Smile                       |
| :------------------ | :-------------------------- |
| CM0001              | O=C(c1ccccc1)CC(=O)c1ccccc1 |

### Target protein data 

`Target_ID` column will be used as foreign key from `Target_ID` from Drug-target interaction data.

| Protein_ID | Sequence                      |
| ---------- | ----------------------------- |
| T0001      | MKRFLFLLLTISLLVMVQIQTGLSGQ... |


## License

CWIDTI follow [GPL 3.0v license](LICENSE). Therefore, CWIDTI is open source and free to use for everyone.

However, drugs which are found by using CWI-DTI follows [CC-BY-NC-4.0](CC-BY-NC-SA-4.0). Thus, those drugs are freely available for academic purpose or individual research, but restricted for commecial use.

## Contact 

liying01@tyut.edu.cn

zhangxingyu0489@link.tyut.edu.cn

wangbin01@tyut.edu.cn
