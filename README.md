## Synopsis

This is neural network for predicting properties of molecules based on their SMILES representation, built on [Keras](https://github.com/fchollet/keras) framework.

## Motivation

Prediction of molecular properties, such as interactions with various proteins, is an important task in drug research and many other fields of chemistry. This project uses recurrent neural networks to read molecules in raw structural format known as SMILES, and is configurable to try learning any given numerical marker.  

## Installation

We use Python 2.7, and the project should run fine on Python 3.5 as well. All dependencies are specified in [requirements](requirements.txt) file.

Training phase significantly benefits from use of GPU. Please consult [Keras](https://github.com/fchollet/keras) documentation for further information about CUDA configuration.

## Execution

Just run `run.py` file, or `gpu.sh` script to run everything. You might want to [configure](#configuration) the neural network first.

## Configuration

This project uses YAML standard for configuration files. List of experiments to run is stated in `local/config.yml`, individual experiment configuration files are located in `local/experiments` folder. 

Config files inherit their default values from templates, which can be found in `config/templates` file, along with documentation for configurable parameters.

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
|   --- config.yml        | Common config specifying list if experiments
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

