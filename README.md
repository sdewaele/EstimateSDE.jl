# EstimateSDE

This is the code implementing the algorithms described in [this paper](https://arxiv.org/abs/2007.01073).

## Accurate Characterization of Non-Uniformly Sampled Time Series using Stochastic Differential Equations

Non-uniform sampling arises when an experimenter does not have full control over
the sampling characteristics of the process under investigation. Moreover, it is
introduced intentionally in algorithms such as Bayesian optimization and compressive
sensing. We argue that Stochastic Differential Equations (SDEs) are especially
well-suited for characterizing second order moments of such time series. We
introduce new initial estimates for the numerical optimization of the likelihood,
based on incremental estimation and initialization from autoregressive models.
Furthermore, we introduce model truncation as a purely data-driven method to reduce
the order of the estimated model based on the SDE likelihood. We show the increased
accuracy achieved with the new estimator in simulation experiments, covering all
challenging circumstances that may be encountered in characterizing a non-uniformly
sampled time series. Finally, we apply the new estimator to experimental rainfall
variability data.

## Citation
You can cite this paper as:
```
@inproceedings{dewaele2020charsde,
  title={Accurate Characterization of Non-Uniformly Sampled Time Series using Stochastic Differential Equations},
  author={De Waele, Stijn},
  booktitle={accepted for the 19th IEEE International Conference On Machine Learning And Applications (ICMLA)},
  year={2020},
  organization={IEEE}
}
```
