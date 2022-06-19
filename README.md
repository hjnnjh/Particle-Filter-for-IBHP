# Particle-Filter-for-IBHP

- This repository provides simulation and the parameter estimator based on Sequential Monte Carlo(SMC) for IBHP.
- `IBHP` was proposed in Paper: [***The Indian Buffet Hawkes Process to Model Evolving Latent Influences***](http://auai.org/uai2018/proceedings/papers/289.pdf).
- **(Important)** Now this repository is still providing an unstable version of implementation of IBHP. There are some mistakes in derivation of model such as likelihood function etc. If you have some suggestions, I'm very glad to receive some pull-requests :)

## Agenda of This Project
    Particle-Filter-for-IBHP
    ├─.gitignore
    ├─LICENSE
    ├─README.md
    ├─requirements.txt
    ├─code
    |  ├─IBHP_simulation_torch.py
    |  ├─particle_filter_torch.py
    |  ├─particle_torch.py
    |  └plot_model_result.py

## Class and Function

### IBHP_simulation_torch.py

In this `.py` file, we provide IBHP simulation implementation based on the algorithm proposed in [Ogata (1981)](https://ieeexplore.ieee.org/abstract/document/1056305)

#### Class

`class IBHPTorch(self,
                 doc_length: int,
                 word_num: int,
                 sum_kernel_num: int,
                 lambda0: torch.Tensor,
                 beta: torch.Tensor,
                 tau: torch.Tensor,
                 n_sample=100,
                 random_seed=None)`

Parameters: 
- `doc_length`: *int*, each document length when simulate IBHP
- `word_num`: *int*, vocabulary size when simulate IBHP
- `sum_kernel_num`: *int*, number of exp kernels in IBHP
- `lambda0`: *torch.Tensor*, value of $\lambda_0$ in IBHP
- `beta`: *torch.Tensor, shape=(sum_kernel_num, )*, value of $\beta$ in IBHP
- `tau`: *torch.Tensor, shape=(sum_kernel_num, )*, value of $\tau$ in IBHP
- `n_sample`: *int, default is 100*, number of observations need to be generated
- `random_seed`: *int, default is None*, like the `random_seed` in `Numpy`, the same data generation results can be reproduced using random number seeds

#### Main Methods:

`generate_data(self, save_result=False, save_path: str = None)`

Generate simulation data.

params:
- `save_result`: *bool, default is False*, whether to save the generated results
- `save_path`: *str, default is None*, path to save the generated results

`plot_intensity_function()`

Plot generated data's intensity $\lambda(t)$, figure will be saved to `./img_test/simulation_data_intensity.png` by default

`plot_simulation_c_matrix()`

Plot generated data's $c$ matrix, figure will be saved to `./img_test/simulation_data_c_matrix.png` by default

### particle_torch.py

In this `.py` file, we implement all the steps of single particle sampling, hyperparameter updating and particle weight calculation.

#### Class

`class Particle(self,
        word_corpus: torch.Tensor,
        timestamp_tensor: torch.Tensor,
        text_tensor: torch.Tensor,
        particle_idx: int,
        sum_kernel_num: int = 3,
        ibhp: IBHPTorch = None,
        fix_w_v: bool = False,
        chunk: bool = False,
        random_seed: int = None,
        device: torch.device = DEVICE0)`

There are many complicated private method in this class, so I choose to skip it. If you are interested in it, just check it in source file, I have written down annotation in code.

`class StatesFixedParticle(self,
                 ibhp: IBHPTorch,
                 particle_idx,
                 word_corpus: torch.Tensor,
                 lambda0: torch.Tensor,
                 beta: torch.Tensor,
                 tau: torch.Tensor,
                 sum_kernel_num: int,
                 random_seed: int = None,
                 device: torch.device = DEVICE0,
                 chunk: bool = False)`

This class is used to test IBHP with fixed particle states (same as the simulation process) such as $c, w, v$.

`class HyperparameterFixedParticle(self,
                 word_corpus: torch.Tensor,
                 particle_idx: int,
                 sum_kernel_num: int = 3,
                 ibhp: IBHPTorch = None,
                 fix_w_v: bool = False,
                 chunk: bool = False,
                 random_seed: int = None,
                 device: torch.device = DEVICE0)`

This class is used to test IBHP with fixed $\lambda_0, \beta, \tau$.

### particle_filter_torch.py

In this `.py` file, we implement particle filter based on the generation process of IBHP.

#### Class

`class ParticleFilter(self,
                 n_particle: int,
                 n_sample: int,
                 word_corpus: torch.Tensor,
                 sum_kernel_num: int,
                 lambda0: torch.Tensor,
                 beta: torch.Tensor,
                 tau: torch.Tensor,
                 alpha_lambda0: torch.Tensor,
                 alpha_beta: torch.Tensor,
                 alpha_tau: torch.Tensor,
                 random_num: int,
                 fix_w_v: bool = False,
                 ibhp_ins: IBHPTorch = None,
                 text_tensor: torch.Tensor = None,
                 timestamp_tensor: torch.Tensor = None,
                 states_fixed: bool = False,
                 hyperparameter_fixed: bool = False,
                 fix_beta: bool = False,
                 fix_tau: bool = False,
                 device: torch.device = DEVICE0,
                 chunk: bool = False)`

Parameters
- `n_particle`: *int*, number of particles used to filtering
- `n_sample`: *int*, total number of observations need to be estimated
- `word_corpus`: *torch.Tensor, shape=(vocab_size, )*, vocabulary tensor
- `sum_kernel_num`: number of exp kernels in IBHP
- `lambda0`: *torch.Tensor*, initial value of $\lambda_0$
- `beta`: *torch.Tensor, shape=(sum_kernel_num, )*, initial value of $\beta$
- `tau`: *torch.Tensor, shape=(sum_kernel_num, )*, initial value of $\tau$
- `alpha_lambda0`, *torch.Tensor*, prior parameter of $\lambda_0$ prior distribution when sampling new model parameter from prior distribution
- `alpha_beta`: *torch.Tensor*, prior parameter of $\beta$ prior distribution when sampling new model parameter from prior distribution
- `alpha_tau`: *torch.Tensor*, prior parameter of $\tau$ prior distribution when sampling new model parameter from prior distribution
- `random_num`: *int*, number of samples when updating model Parameters
- `fix_w_v`: *(bool, optional), Defaults to False.*, used in model test, let $w$ and $v$ same as simulated values
- `ibhp_ins`: *(IBHPTorch, optional), Defaults to None.*, used in model test, `IBHPTorch` instance, read fixed value from it (such as $w$ and $v$)
- `text_tensor` *(torch.Tensor, shape=(n_sample, vocab_size), optional). Defaults to None.*, if in test, this is unnecessary.
- `timestamp_tensor` *(torch.Tensor, shape=(n_sample, ), optional). Defaults to None*, if in test, this is unnecessary.
- `states_fixed` *(bool, optional). Defaults to False*, if `True`, use `StatesFixedParticle`
- `hyperparameter_fixed` *(bool, optional) Defaults to False*, if `True`, use `HyperparameterFixedParticle`
- `fix_beta` *(bool, optional). Defaults to False*, if `True`, set $\beta$ same as simulated value
- `fix_tau` *(bool, optional). Defaults to False*, if `True`, set $\tau$ same as simulated value
- `device` *(torch.device, optional). Defaults to DEVICE0*, controls on which devices the model is estimated
- `chunk` *(bool, optional). Defaults to False*, when `n_sample` and `random_num` is large, set this param to `True` to avoid GPU memory overflow.

#### Main Methods

`filtering(self,
                  save_dir: str,
                  prefix: str = 'test',
                  rename_by_timestamp: bool = False,
                  save_res: bool = False):`

Parameters:
- `save_dir`: *(str)*, path to save filtering result
- `prefix`: *(str, optional)*, prefix in save path
- `rename_by_timestamp`: *(bool, optional)*, whether rename folder by adding timestamp behind prefix
- `save_res`: *(bool, optional)*, whether save filtering result

### plot_model_result.py

I provide a few functions to plot model result, if you are interested in them, please check the source code.

## Usage

I wrote down a simple demo in the end of `particle_filter_torch.py`, please check it.
