# Bayesian Zero-Shot Learning

Matlab implementation of our "Bayesian Zero-Shot Learning" paper. Accepted to *ECCV 2020, TASK-CV Workshop*.
Authors: *Sarkhan Badirli, Zeynep Akata, and Murat Dundar*

Paper at: https://arxiv.org/pdf/1907.09624.pdf

<p align="center">
  <img width="400" height="400" src="Process_final.png">
</p>
<p align="justify">

## Brief Summary

We propose a hierarchical Bayesian model based on the intuition that actual classes originate from their corresponding local priors, each defined by a meta-class of its own. We derive the posterior predictive distribution (PPD) for a two-layer Gaussian mixture model to effectively blend local and global priors with data likelihood. These PPDs are used to implement a maximum-likelihood classifier, which represents seen classes by their own PPDs and unseen classes by meta-class PPDs. Across seven datasets with varying granularity and sizes, in particular on the large scale ImageNet dataset, we show that the proposed model is highly competitive against existing inductive techniques in the GZSL setting.


## Prerequisites

The code was implemented in Matlab. Any version greater 2016 should be fine to run the code.

## Data

You can download the datasets used in the paper from [Google Drive](https://drive.google.com/file/d/1BjTar-w9ysiHi1N4RpNguCYOONSaDsSP/view?usp=sharing).  Create a `data` folder in your main `project path` and put the data under this folder.


## Experiments

To reproduce the results from the paper, open the `Demo.m` script and specify the dataset and model version (*unconstrained* or *constrained*). Please change the datapath to your project path in `Demo.m` script.

If you want to perform hyperparameter tuning, please comment out relevant sections from `Demo.m` script.

The results may vary 1-2% or less between identical runs in *constrained* model due to random initialization.

### Contact

Feel free to drop me an email if you have any questions: s.badirli@gmail.com
 
