# MGP-TCN for Sepsis Prediction on the MIMIC Dataset

This repository comprises code concerning sepsis prediction (code for labelling, extraction and proposed method) of the Paper: 

```
@InProceedings{moor2019early,
  title		=	{Early Recognition of Sepsis with Gaussian Process Temporal Convolutional Networks and Dynamic Time Warping},
  author	=	{Moor, Michael and Horn, Max and Rieck, Bastian and Roqueiro, Damian and Borgwardt, Karsten},
  booktitle	=	{Proceedings of the 4th Machine Learning for Healthcare Conference},
  pages		=	{2--26},
  series        = 	{Proceedings of Machine Learning Research},
  publisher     = 	{PMLR},
  year		=	{2019}
}
```

[Moor, M., Horn, M., Rieck, B., Roqueiro, D., & Borgwardt, K. (2019). Early Recognition of Sepsis with Gaussian Process Temporal Convolutional Networks and Dynamic Time Warping.](http://proceedings.mlr.press/v106/moor19a/moor19a.pdf) 

## Quick start on simulated data

To get a quick start of our proposed method MGP-TCN (Multi-task Gaussian Process Adapted Temporal Convolutional Networks) 
with simulated data, simply run:

    $ make mock_data_results

from the main directory of this repository. It runs the MGP-TCN method on
simulated health records data without the need to get access to,
download, install, and query the MIMIC database.

You should receive the following output (after some initial updates):

    >>>>>> 656128
    >>>>>> Starting epoch 0
    >>>>>> Batch 0/45, took: 3.349, loss: 3575.63257
    >>>>>> Batch 1/45, took: 1.005, loss: 2463.76196
    >>>>>> Batch 2/45, took: 1.076, loss: 2280.01270

To actually run our method on the MIMIC sepsis cohort (or generally use this cohort due to its high-resolution sepsis-3 label), 
follow the full instructions below (which as a warning will take several hours and considerably longer than just running the test script above).

Throughout this repo, you can find instructive textfiles named 'info.txt'. In there you find information on how to run the scripts in the corresponding directory (e.g. query etc.).

The codebase is structured in three sequential steps:
 1. Querying the MIMIC relational database to compute the sepsis label and extract input data
 2. Preprocessing Steps
 3. Experiments 


## Query and real-world experiments

This repository provides a postgresql-pipeline to extract vital time series of sepsis cases and controls from the MIMIC database following the recent SEPSIS-3 definition.
 
1. Requirements for MIMIC:
  a) Requesting Access to MIMIC (publicly available, however with permission procedure)
      https://mimic.physionet.org/gettingstarted/access/
  b) Downloading and installing the MIMIC database according to documentation: 
      https://mimic.physionet.org/gettingstarted/dbsetup/  
      unix/max: https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/  
      windows: https://mimic.physionet.org/tutorials/install-mimic-locally-windows/ 

2. Run the pipeline:
    a) Once the requirements are fulfilled, open the Makefile and
        - specify your username and database name
    
    To run the query, type:

        $ make query

    b) For running the entire pipeline (including query and a mgp-tcn experiment), simply run:

        $ make

    c) Given the query was executed before and you wish to execute an mgp-tcn experiment (as configured in configs), use:

        $ make run_experiments    
    
    d) For running an experiment with the exact hyperparameters we found in our hyperparameter search, use one of the configs in paper_configs like this:
        
        $ python3 -m src.mgp_tcn.mgp_tcn_fit with configs/paper_configs/config_split0.json

    Note: 
	- This step might take several hours as the rather involved query is highly relational and therefore easier to perform in postgresql rather than processing all data in a dynamic environment.
        - A faster, more dynamic implementation outside of postgres would be favorable, however most of our queries are extensions of the mimic-code repository (in postgres)


## Library Versions (which were used for development)
If you cannot by default run the above code, make sure your libraries fit our used versions:

for the experiments a GPU-based tensorflow environment is assumed. For packages see requirements.txt
and run:

    $ pip install -r requirements.txt

for the queries, make sure you follow the mimic tutorial instructions for setting it up. We used
- PostgreSQL 9.3.22 on x86_64-unknown-linux-gnu, compiled by gcc (Ubuntu 4.8.4-2ubuntu1~14.04.4) 4.8.4, 64-bit

## Acknowledgements
- Contributors of this repo: Michael Moor, Max Horn, Damian Roqueiro.
Please cite our corresponding paper: 
[Moor, M., Horn, M., Rieck, B., Roqueiro, D., & Borgwardt, K. (2019). Early Recognition of Sepsis with Gaussian Process Temporal Convolutional Networks and Dynamic Time Warping.](http://proceedings.mlr.press/v106/moor19a/moor19a.pdf) 

Depending on which part of the codebase you use, please also cite our main sources:

- A large part of the query code was inspired by this project: https://github.com/alistairewj/sepsis3-mimic. Some of their code reappears here modified, some basic scripts even untouched. A big thanks to their valuable contribution.
[Johnson, Alistair EW, David J. Stone, Leo A. Celi, and Tom J. Pollard. 
"The MIMIC Code Repository: enabling reproducibility in critical care research." 
Journal of the American Medical Informatics Association (2017): ocx084.](https://www.ncbi.nlm.nih.gov/pubmed/29036464)

- We used Futoma et. al's differentiable tensorflow implementation of the MGP as proposed in this paper.
[Futoma, Joseph, Sanjay Hariharan, and Katherine Heller. "Learning to detect sepsis with a multitask Gaussian process RNN classifier." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.](https://arxiv.org/abs/1706.04152)




