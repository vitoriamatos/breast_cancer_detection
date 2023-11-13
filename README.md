# breast_cancer_detection

This project uses 4 neural network structures to train the recognition of cancerous lesions, using mammographic and thermographic images as a database. As results, metrics relating to the training of each neural network are obtained.

## Requirements
For this project, we recommend using Nvida's CUDNN and ANACONDA.

## Instalation
You also need to run this script to set up a directory structure and install dependencies:

```bash
chmod +x setup_script.sh
```

```bash
./setup_script.sh
```

## Database
After downloading the respective databases, we suggest that you extract the files into the directories of each database contained in  `/databases`


* Mias Mammograph | Donwload

    * After downloading the database, extract its contents into the `/databasesmias_database` direrectory

* Mammotherm | Donwload

    * After downloading the database, extract its contents into the `/mammotherm_database` direrectory 



## Run Application
Once this is done, and after completing the installation of the dependencies, you can choose which database to train:

```bash
    python train_mias.py
    pyton train_mammotherm.py
```

