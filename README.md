## Synopsis

This project implements recurrent neural network with ultimate goal of predicting edge property of SMILES reperesented molecule, and FASTA represented protein. Powered by [Keras](https://github.com/fchollet/keras) framework.

## Motivation

Prediction of molecular properties, such as interactions with various proteins, is an important task in drug research and many other fields of chemistry. This project uses recurrent neural networks in order to read molecular and protein structure in ASCII based formats known as SMILES or FASTA, and learn to predict properties of interaction between these compounds.

## Installation

The application is developed and tested on Python 2.7. Required dependencies are specified in `requirements.txt` file, and should be available using `pip`. We use [Theano](https://github.com/Theano/Theano) as a backend for Keras, TensorFlow option is not tested.

Training phase run times significantly benefit from use of GPU. Please consult [Keras](https://github.com/fchollet/keras) documentation for further information about CUDA configuration.

## Execution

Just run `run.py` file, or `gpu.sh` script to run everything. You might want to [configure](#configuration) the neural network first.

## Configuration

This project uses YAML standard for configuration files. List of experiments to run is stated in `local/config.yml`, individual experiment configuration files are located in `local/experiments` folder. 

Config files inherit default values of parameters from templates, which can be found in `config/templates` folder, along with documentation for configurable parameters.

## Project structure

List of key project files and directories:

```
.
+-- baselines             | Standalone non-NN baseline models for comparison
+-- computing             | Visualization and statistics of datasets
+-- config                
|   +-- templates         | Templates of default values for configs
|   --- config.yml        | Common configuration for all experiments
+-- db                    | Remote DB connection
+-- local
|   +-- experiments       | Custom experiment configs
|   --- config.yml        | Root config specifying list of experiments to be executed
+-- rnn
|   --- data.py           | Data preprocessing
|   --- dnn.py            | Deep Neural Network model essentials
|   --- metrics.py        | Statistics computation
|   --- rnn.py            | Recurrent Neural Network model essentials
|   --- utility.py        | Miscellaneous methods
|   --- visualization.py  | Data and results visualizations
--- gpu.sh                | Script for running app under GPU
--- requirements.txt      | Project dependencies
--- run.py                | Main run script
```

## License

This project is licensed under [BSD-3-Clause License](LICENSE).

