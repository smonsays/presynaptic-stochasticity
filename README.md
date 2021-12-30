# Presynaptic stochasticity & plasticity

This repository contains code to execute the main experiments of the paper [Presynaptic stochasticity improves energy efficiency and helps alleviate the stability-plasticity dilemma](https://doi.org/10.7554/eLife.69884).

## Usage

### Experiments

- Energy experiments can be run by executing the bash script `exp_energy.sh`
- Lifelong experiments can be run by executing the bash script `exp_lifelong.sh` and the Jupyter notebook `run_lifelong-perceptron.ipynb`
- Ablation experiments can be run by executing the bash scripts `exp_ablation_split-mnist.sh` and `exp_ablation_perm-mnist.sh`

### Main model

The main model can be experimented with using `python run_dyn_continual.py`. Use the `-h` flag to show the various options.

## Requirements

Code was tested using `python 3.9`. Required packages can be found in `requirements.txt` and installed using

```
pip install -r requirements.txt`
```

## Project structure

- `./` Contains python scripts to run the models and bash scripts to automate experiments, most notably:
  - `run_dyn_continual.py` runs the main model for various configurations
- `lib/` Contains the modules implementing the core functionality of the algorithm, most notably:
  - `ddc.py`: Contains the main logic of the algorithm
  - `train.py`: Contains the training routine
- `etc/` Contains default hyperparameter configurations for different tasks
- `log/` Default logging directory

## Citation

If you use this code in a scientific publication, please include the following reference in your bibliography:

```bibtex
@article {10.7554/eLife.69884,
article_type = {journal},
title = {Presynaptic stochasticity improves energy efficiency and helps alleviate the stability-plasticity dilemma},
author = {Schug, Simon and Benzing, Frederik and Steger, Angelika},
editor = {Behrens, Timothy E and O'Leary, Timothy and Pfister, Jean-Pascal},
volume = 10,
year = 2021,
month = {oct},
pub_date = {2021-10-18},
pages = {e69884},
citation = {eLife 2021;10:e69884},
doi = {10.7554/eLife.69884},
url = {https://doi.org/10.7554/eLife.69884},
journal = {eLife},
issn = {2050-084X},
publisher = {eLife Sciences Publications, Ltd},
}
```

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/) - see the LICENSE file for details.

## Code Style

This project uses `flake8` for linting and adheres to the [pep8](https://pep8.org/) standard.
