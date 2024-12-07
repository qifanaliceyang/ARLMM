import numpy as np
import pandas as pd
import scipy as sp
import sys, pickle
import time
import scipy.stats
import matplotlib
from scipy.stats import multivariate_normal
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
#from utils import mapping_matrix_T, cross_sectional_longit_mapping_function,load_groups_from_file, load_random_effect_info_from_file, project_covariate_func, ARLMM 
#from utils import predict, evaluate, booostrap_stats, group_kfold_cross_validation, final_calculation 
from ARLMM import ARLMM_Model
from simulation import GenomeSimulator, LongitudinalPhenotypeSimulator, CovarianceSimulator


class TypeIErrorCalculator:
    """
    A class to calculate Type I error rates for a given statistical model.
    """


class PowerCalculator:
    """
    A class to calculate statistical power for a given model and effect size.
    """




def main():
    # Parameters for the simulation
    n_loci = 1500  # Number of loci (genetic markers)
    n_simulations = 5  # Number of Simulations
    n_parameters = 7  # Number of parameters to estimate
    maf_range = (0.05, 0.4)  # SNP MAF range
    baseline_mean = 3000  # Mean baseline brain volume (phenotype)
    baseline_sd = 100  # Standard deviation of the baseline phenotype
    noise_sd = 50  # Standard deviation of random noise
    ar1_rho = 0.8  # Autoregressive parameter for AR(1)
    covar_dynamic_name_list = ['Age', 'Sex', 'ICV']  # Covariates

    # Vary the number of subjects (individuals) to observe the impact on running time
    subject_numbers = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]  # Number of subjects to test


if __name__ == "__main__":
    main()


