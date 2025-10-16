# README
This README would normally document whatever steps are necessary to get your application up and running.

## What is this repository for?
This repository contains source code for heirarichal classifer using methylation data. Code is being migrated from a general classifiers/ Version 0.1

## Set up
There should be set up instructions, yes.

scripts to find and generate data are in /data_processing. 
sample_file_generation.py generates files for new samples, it is currently run as a cron job nightly. Generated files will contain all probes for each sample, and are the base data file that is used as inputs to the filtering steps prior to model generation. sample_feature_collection selects the features that were identified as part of model training and adds them to a dataframe that is used to serve results.

- Summary of set up
- Configuration

Paths to credential files the config directory will need to be altered.
Paths for the base file locations will need to be updated in base_config.yaml

Dependencies Built and run using python 3.11 packages required are to be found in requirements.txt - this probably has packages over and above what is required. i.e. it could do with being cleaned.

Configuration. Currently, there are 3 places that one has to define which dataset to work on. I am currently using a variable called freeze, to define location. i.e. : /\<base data direcotry>/freeze\<monthyear>/ This needs to configured in the files

- mch.config.settings
- mch.core.create_disease_tree
- mch.data_processing.dataset_filtering I need to change this. At the moment, most other parts of this will pull from the settings. create_disease_tree and dataset_filtering have to be run prior to settings being able to work though

In later iterations, there should also probably be the ability to pass config files as part of the command line. Or at least a base directory. That's for when code is to be released/someone else needs to run it on a different system.

Database configuration In /db there are connectors to the sql database and to type db. These will need credentials. If we move this to databricks then I am assuming there will need to be additional code to mange access to the data.

How to run tests I do actually have some pytest tests. Not all of them run sucessfully as of yet.

Deployment instructions

## Contribution guidelines
- Writing tests
- Code review
- Other guidelines

##Who do I talk to?
Ben
