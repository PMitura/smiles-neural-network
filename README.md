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

TODO

## Contributors

TODO

## License

This project is licensed under [BSD-3-Clause License](LICENSE).

## How to contribute

TODO
