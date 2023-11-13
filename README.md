# breast_cancer_detection

This project uses 4 neural network structures to train the recognition of cancerous lesions, using mammographic and thermographic images as a database. As results, metrics relating to the training of each neural network are obtained.

## Requirements
For this project, we recommend using Nvida's CUDNN and ANACONDA.

## Database
After downloading the respective databases, we suggest you create the database directory, if you are using Linux you can follow these steps:

```bash
    mkdir databases
```

```bash
    cd databases/
```
```bash
    mkdir mammotherm_database
```
```bash
    mkdir mias_database
```
* Mias Mammograph | Donwload

    * After downloading the database, extract its contents into the mias_database directory

* Mammotherm | Donwload

    * After downloading the database, extract its contents into the mammotherm_database directory


## Instalation
First install the dependencies:
```bash
pip install -r requirements.txt
```
During the installation process you can create directories to store metrics and models

* Models
    ```bash
        mkdir models
    ```

    ```bash
        cd models/
    ```
    ```bash
        mkdir mammotherm
    ```
    ```bash
        mkdir mias
    ```


* Methrics
    ```bash
        mkdir methrics
    ```

    ```bash
        cd methrics/
    ```
    ```bash
        mkdir mammotherm
    ```
    ```bash
        mkdir mias
    ```
Once this is done, and after completing the installation of the dependencies, you can choose which database to train:

```bash
    python train_mias.py
    pyton train_mammotherm.py
```

