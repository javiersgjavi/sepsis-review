[![DOI:10.1007/s10489-023-05124-z](http://img.shields.io/badge/DOI-10.1007/s10489--023--05124--z-32a852.svg)](https://doi.org/10.1007/s10489-023-05124-z) [![DOI](https://zenodo.org/badge/567043441.svg)](https://zenodo.org/doi/10.5281/zenodo.10203161)

#   [Comparing artificial intelligence strategies for early sepsis detection in the ICU: an experimental study](https://doi.org/10.1007/s10489-023-05124-z)

This repository contains the code used in the paper with the same title.


# Table of Contents
1. [Abstract](#Abstract) 
2. [Authors](#Authors) 
3. [Prerequisites](#Prerequisites)
4. [How to download and generate the data](#Generate)
5. [What does the repository?](#What_Does)
6. [How to define a parameter optimization and an experiment?](#Define)
7. [How to launch a parameter optimization and an experiment?](#Launch)
8. [How works the code of the repository?](#Code)
9. [License](#license) 

<a name="Abstract"/>

## Abstract

[An Experimental Review on the Early Prediction of Sepsis in the
ICU](doi)

> Sepsis is a life-threatening organ dysfunction due to a dysregulated host response to infection and constitutes a major global health concern. Hence,
we need a systems medicine approach to face one of the challenges in sepsis
disease: early recognition in intensive care units (ICU). In this scenario, machine learning algorithms can be used to extract patterns from the amount
of data. The goal is to mine and exploit health data using machine learning because an amount of clinical data (vital signs, medications, laboratory
measurements, etc.) and health history are available in databases for patients suffering from sepsis. Recently, several works face this challenge, but
a few of them provided the code and it is very difficult to compare the results. Therefore, this work performed an experimental review of publications
discussing early prediction of sepsis in the ICU using machine learning algorithms. Briefly, we reviewed the literature, we analyzed several imputation
strategies because clinical data is commonly sampled irregularly, requiring
a set of hand-crafted preprocessing steps, and we conducted an extensive
experimental study using five classical machine learning methods and five
popular deep learning models using an offline training with horizon evaluation. Finally, deep learning methods (TCN and LSTM) outperform the
other methods by a significant margin, especially for early detection tasks
more than 4 hours before the onset of sepsis, while Random Forest shows the
worst behavior.


<a name="Authors"/>

## Authors:
- Javier Solís-García
- Belén Vega-Márquez
- Juan Nepomuceno
- José C. Riquelme-Santos
- Isabel A. Nepomuceno-Chamorro


<a name="Prerequisites"/>

## Prerequisites

This repository has been tested with the following requirements; however, it may be run with a different version of the listed software:

1. Ubuntu 20.04 or 22.04
2. Docker version 20.10.18
3. docker-compose version 1.29.2
4. Nvidia Container Tookit

<a name="Generate"/>

## How to download de data

### Clone the repository

The repository can be cloned with the command: ```git clone https://github.com/javiersgjavi/sepsis-review.git```


### _How to download the Physionet challenge data?_:

This dataset is only used in the Appendix of the article. It is available from [Physionet Challenge website](https://physionet.org/content/challenge-2019/1.0.0/). However, in this repository we have downloaded the data available in Kaggle [Kaggle](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis) which already mix the data from the two different hospitals.

1. Make sure you have installed the Kaggle API: ```pip install kaggle```.
2. Give execution files to the file _download_data.sh_ with the command: ```chmod +x download_data.sh```.
3. Execute the file _download_data.sh_ with the command: ```./download_data.sh```.

### _How to download MIMIC-III data_:

_This is the dataset which is mainly used in the article_

### Generate the data

```INFO:```: ***If you are only interested in obtaining the data, I recommend you to visit this other [repository](https://github.com/javiersgjavi/tabular-mimic-iii)***

```WARNING:``` This guide has been made for Ubuntu 20.04, it should be similar for other Linux versions, but may differ for a different operating system

***If you are only interested in obtaining the data, I recommend you to visit this other repository***

#### Download data from PhysioNet and create PostgresSQL DB with the [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code) repository

1. Request access to MIMIC-III data in [PhysioNet](https://mimic.mit.edu/docs/gettingstarted/).

2. Go to the folder _build_mimic_data_: ```cd build_mimic_data```.

3. Add your user of physionet with access privileges to MIMIC-III in line1 of file _download_physionet_data.sh_, where it says <User_of_physionet>.

4. Give execution permissions to all the .sh files with the following command: ```chmod +x *.sh```.

5. Execute the following command: ```sudo ./download_physionet_data.sh```. The process can be long and will require:

- Enter the password of your PhysioNet account to download the data.
- You will watch the log of the postgres database being loaded. You must wait until you see a table with the tables of the DB where all rows return "PASSED", and below must display the message "Done!" before going to the next steps.
- Once the process is finished, press: ```Ctrl + C``` to finish the display of the log.

#### Create CSV files from the PostgresSQL DB with the [BorgwardtLab/mgp-tcn](https://github.com/BorgwardtLab/mgp-tcn) repository
6. Execute the following command: ```./preprocess_data.sh```.
7. Give execution permissions _main.sh_ file with the command: ```chmod +x src/query/main.sh```.
8. Execute the command: ```make query```. This process will be longer than the one in the 6 step.

#### Create the final data with [BorgwardtLab/mgp-tcn](https://github.com/BorgwardtLab/mgp-tcn) repository
9. Execute the command: make generate_data to generate the final data that the repository will use.
10. Exit from the container with the command: ```exit```.
11. Execute the command: ```./create_data_folder.sh```.

<a name="What_Does"/>

## What does the repository?

This repository focuses on early prediction of sepsis onset. To archive this task, the repository is divided into two main sections:
- Parameter Optimization: In this part of the code, the mission is to find the best combination of parameters and imputation method to classify patients who will have an onset of sepsis with data from the 49 hours prior to sepsis onset for each model tested. In addition, this will create the data to make an experimental comparison between different imputation techniques and models.

- Experiment: Once the optimization of the parameters is done, with the best configuration for each model, an experiment will be performed in which different time horizons before the onset of sepsis will be tested to check the performance of each model.


<a name="Define"/>

## How to define a parameter optimization and an experiment?

The characteristics of the parameter optimization and the experiments are defined in the file __main.py__. these are some important variables:

- **name**: defines the name of the experiment and the name of the folder with the results.

- **iterations_sampler**: defines the max number of different configurations of hiper-parameters that are tested for each model. Is used to reduce the execution time by reducing the total amount of models tested. For more exhaustive experimentation, this number must be increased.

- **models** : there is one variable for each task, which defines the models that are going to be used. The variable is a dict with all the models implemented for the task, if you want to enable any of them, you have to put 1 as the value of the key. To deactivate the model, you have to put 0 as the value.

- **imputation_methods**: this one is also similar to the last two. It is a dict with all available imputation methods, put 1 to activate an imputation method, or 0 to deactivate it.

- **data**: define the name of the data that is going to be use.

On the other hand, the main variable of the experiment is:

- **hours_before_onset**: During the experiment, models will be tested with different horizons, ranging from 1 hour before sepsis onset to the number of hours before onset defined in this variable.

<a name="Launch"/>

## How to launch a parameter optimization and an experiment?

- It is important to give execution permissions to the _start.sh__ file if it hasn't with the command: ```chmod +x start.sh```.

- Start and enter to the docker container with the command: ```./start.sh```. 

- Launch the experiment with the command: ```python main.py```. 

- If you want to execute in second plane, use the command: ```nohup python main.py```.

<a name="Code"/>

## How works the code of the repository?

- **main.py**: it is the script that defines and launches a parameter optimization and an experiment.
- **src/utils/generate_reports.py**: it is a script that generates all tables and images with the summary of the parameter optimization and experiment results.
- **src/utils/preprocess_data.py**: it is a script that contains all functions that preprocess the data. This script contains all functions to make the different imputations methods and probably is the more messy and difficult file to understand in this repository. I'm sorry for the untidy state of this file, but the management of the input data was very chaotic due to its format.
- **src/classes/ParameterOptimization.py**: is one of the main class of this repository. Prepare de data, execute the models and save the results.
- **src/classes/Experiment.py**: is the other main class in this repository. It loads each model with the best parameters and the best imputation method for the data found during the optimization, and tests each model with different horizons, which will start from one hour before start to as many hours before start as defined in main.py.
- **src/classes/Data.py**: this file contains classes related to the management of the data, the normalization, and the hyperparameter values for the models. 
- **src/classes/DL.py**: this file contains the Deep Learning models implemented in this project, whose models extend the class DL. The addition of a new model is very modular and only must extend the DL class, similar to the other ones.
- **src/classes/ML.py**: this file contains the Machine Learning models implemented in this project, whose models extend the class ML. The addition of a new model is also modular and only must extend the ML class, similar to the other ones.
- **src/classes/Metrics.py**: this file contains the metrics implemented in this project. To add a new metric, it must be added as a method in the MetricCalculator class.
- **docker/**: this folder contains the files to create the docker image and the docker container.
- **parameters.json**: this file contains the possible values for the parameters of each model.
- **data**: this folder will be generated during the execution of the repository and will contain the original data and the imputed data to reduce the execution time.
- **results**: this folder will be generated during the execution of the repository too. It will contain the tables, images, and predictions of the results.csv which is the file that contains all the data of the parameter optimization and the experiment.
- **build_mimic_data**: this folder is used to download and generate the data which will be used. It contains the clone of two repositories:
  - [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code) repository: is used to download de MIMIC-III data from physionet
  - [BorgwardtLab/mgp-tcn](https://github.com/BorgwardtLab/mgp-tcn): is used to generate the data that will be used by the experiments.


## License<a name="license"></a>

This project is licensed under the BSD-3-Clause license - see the [LICENSE](LICENSE) file for details
