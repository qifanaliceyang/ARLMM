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
    def __init__(self, n_individuals, repeat_measures, n_simulations):
        """
        Initializes the power analysis class for longitudinal data with random effects.

        n_individuals: Number of individuals
        repeat_measures: List specifying the number of repeated measures for each individual
        n_simulations: Number of simulations to run for the power analysis
        """
        self.n_individuals = n_individuals
        self.repeat_measures = repeat_measures
        self.n_simulations = n_simulations

    def fit_mixed_model(self, data):
        """
        Fits a linear mixed model to the simulated data.

        param data: DataFrame with columns: 'Individual', 'Time', and 'Phenotype'
        return: p-value for the fixed slope effect
        """
        result = model.fit() # did we implement model.fit()?
        p_value = result.pvalues
        return p_value

    def perform_power_analysis(self, method):
        """
        Performs power analysis by simulating data multiple times and computing the power (p values):

        return: Estimated power (proportion of simulations where p-value < alpha)
        """
        significant_results = 0

        for sim in range(self.n_simulations):
            # Simulate data
            simulated_data = self.simulate_data()

            # Fit the mixed model and get the p-value for SNP
            p_value = self.fit_mixed_model(simulated_data)

            # Check if the result is statistically significant
            if p_value < self.alpha:
                significant_results += 1

        # Compute power (proportion of simulations where p-value < alpha)
        power = significant_results / self.n_simulations
        return power



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


