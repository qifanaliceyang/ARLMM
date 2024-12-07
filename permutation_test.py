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
from Simulation import GenomeSimulator, LongitudinalPhenotypeSimulator, CovarianceSimulator


if __name__ == "__main__":
    main()


class PermutationLMMTest:
    def __init__(self, arlmm_model, Y, X, GRM, measurement, subject_level, n_permutations=1000):
        """
        Initializes the permutation test for ARLMM Model in longitudinal data.

        Parameters:
        arlmm_model (ARLMM_Model): Instance of the ARLMM_Model class.
        Y (array): Longitudinal data ordered by time increasing per subject.
        X (array): Covariate matrix.
        GRM (array): Genetic relationship matrix for random effects.
        measurement (list): Number of measurements for each subject.
        subject_level (array): Cohort ID for each subject.
        n_permutations (int): Number of permutations.
        """
        self.arlmm_model = arlmm_model  # Instance of ARLMM_Model
        self.Y = Y
        self.X = X
        self.GRM = GRM
        self.measurement = measurement
        self.subject_level = subject_level
        self.n_permutations = n_permutations
        self.original_stat = None
        self.permuted_stats = []

    def fit_original(self):
        """
        Fit the original ARLMM model to the longitudinal data and extract the test statistic.
        """
        # Fit the ARLMM model with the original data
        self.arlmm_model.fit(self.Y, self.X, self.GRM, self.measurement, self.subject_level)

        # Store the original test statistic (e.g., heritability or fixed effect coefficient)
        self.original_stat = self.arlmm_model.h2_est  # or other statistics such as beta coefficient
        print(f"Original statistic (Heritability h2): {self.original_stat}")

    def permute_within_subject(self):
        """
        Permute the outcome Y within each subject, maintaining the longitudinal structure.
        """
        permuted_Y = self.Y.copy()
        start = 0
        for n_meas in self.measurement:
            # Shuffle Y values within each subject's time-series data
            permuted_Y[start:start + n_meas] = shuffle(permuted_Y[start:start + n_meas])
            start += n_meas
        return permuted_Y

    def fit_permuted(self):
        """
        Perform the permutation test by shuffling Y and refitting the model.
        """
        for i in range(self.n_permutations):
            # Permute within subjects (preserving longitudinal structure)
            permuted_Y = self.permute_within_subject()

            # Fit the ARLMM model on permuted Y
            self.arlmm_model.fit(permuted_Y, self.X, self.GRM, self.measurement, self.subject_level)

            # Store the permuted test statistic (e.g., heritability or fixed effect coefficient)
            permuted_stat = self.arlmm_model.h2_est  # or use other statistics, e.g., beta coefficient
            self.permuted_stats.append(permuted_stat)

            # Optionally print progress every 100 permutations
            if (i + 1) % 100 == 0:
                print(f"Permutation {i + 1}/{self.n_permutations}, Permuted statistic: {permuted_stat}")

    def calculate_p_value(self):
        """
        Calculate the p-value based on the permuted test statistics.

        Returns:
        float: The p-value for the original statistic.
        """
        if self.original_stat is None:
            raise ValueError("Original model has not been fitted. Call fit_original() first.")

        # Calculate p-value: proportion of permuted stats greater than or equal to the original
        permuted_stats = np.array(self.permuted_stats)
        p_value = np.mean(np.abs(permuted_stats) >= np.abs(self.original_stat))

        print(f"P-value from permutation test: {p_value}")
        return p_value
    def visualize_permuted_results(self, permuted_results):
        """Visualize the distribution of permuted estimates and compare to the original."""
        estimates = ['rho_est', 'var_g_est', 'var_t_est', 'var_c_est', 'var_e_est', 'h2_est', 'beta_est']
        original_estimates = [
            self.original_rho_est, self.original_var_g_est, self.original_var_t_est, 
            self.original_var_c_est, self.original_var_e_est, self.original_h2_est, self.original_beta_est
        ]

        for i, estimate in enumerate(estimates):
            if estimate != 'var_c_est' or self.original_var_c_est != 'N/A':  # Skip var_c_est if it is N/A
                plt.figure(figsize=(8, 5))
                sns.histplot(permuted_results[estimate], kde=True, bins=30, color='skyblue', label='Permuted Estimates')
                plt.axvline(original_estimates[i], color='red', linestyle='--', label=f'Original {estimate}')
                plt.title(f'Distribution of Permuted Estimates for {estimate}')
                plt.xlabel(f'{estimate}')
                plt.ylabel('Frequency')
                plt.legend()
                plt.show()


def run_permutation_test(n_individuals, n_loci, maf_range, repeat_measures, baseline_mean, baseline_sd, noise_sd, ar1_rho, covar_dynamic_name_list, n_simulations):
    simulation_p_value_results = []

    # Loop through each simulation
    for i in range(n_simulations):
        # Initialize genome simulator
        simulator = GenomeSimulator(n_individuals, n_loci, maf_range, population_structure=False, n_subpopulations=2)

        # Simulate genotype data and fixed SNP effects
        genotype_data = simulator.simulate_genotype_data()
        # Add Principle Components to covariates
        principal_components = simulator.compute_principal_components(n_components=4, plot=True)

        SNP_fixed_effect = simulator.simulate_fixed_snp()

        # Compute Genetic Relationship Matrix (GRM)
        grm = simulator.compute_grm()

        # Initialize phenotype simulator
        phenotype_simulator = LongitudinalPhenotypeSimulator(n_individuals, repeat_measures, baseline_mean, baseline_sd, noise_sd, ar1_rho)

        # Simulate autoregressive longitudinal phenotypes
        autoregressive_phenotypes = phenotype_simulator.simulate_autoregressive_phenotypes()
        autoregressive_phenotypes = np.expand_dims(autoregressive_phenotypes, axis=1)

        # Initialize covariance simulator
        covariance_simulator = CovarianceSimulator(n_individuals, repeat_measures)
        df = covariance_simulator.simulate_longitudinal_data()

        # Extract covariance data (including covariance/PCs/SNP)
        covariance_simulator_data = np.array(df[covar_dynamic_name_list])


        # Fit ARLMM Model with simulated data
        model = ARLMM_Model()

        perm_test = PermutationLMMTest(model, autoregressive_phenotypes, covariance_simulator_data, grm, repeat_measures, subject_level=None, n_permutations=500)

        perm_test.fit_original()
        perm_test.fit_permuted()
        simulation_p_value_results.append(perm_test.calculate_p_value())

        # Store the results in the results list
        #model.var_c_est = model.var_c_est if hasattr(model, 'var_c_est') else 'N/A'
        #simulation_results.append([model.rho_est, model.var_g_est, model.var_t_est, model.var_c_est, model.var_e_est, model.h2_est, model.beta_est])

    # Calculate the average time for all simulations
    # average_time = total_time / n_simulations
    # return average_time, simulation_results
    return np.mean(simulation_p_value_results)


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
    #subject_numbers = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # Number of subjects to test
    subject_numbers = [100]
    average_times = []
    average_memory_usages = []

    for n_individuals in subject_numbers:
        repeat_measures = np.repeat(3, n_individuals)  # Repeat measures for each individual
        print(f"Running simulation for {n_individuals} individuals...")

        # Run the simulation and get the average running time
        p_value = run_permutation_test(n_individuals, n_loci, maf_range, repeat_measures, baseline_mean, baseline_sd, noise_sd, ar1_rho, covar_dynamic_name_list, n_simulations)