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


class GenomeSimulator:
    def __init__(self, n_individuals, n_loci, maf_range=(0.05, 0.5), population_structure=False, n_subpopulations=2):
        """
        Initializes the class with the number of individuals, loci, and other parameters (cross sectional).

        n_individuals: Number of individuals
        n_loci: Number of loci (genetic markers)
        maf_range: Tuple (min_maf, max_maf) specifying the range of MAF to simulate
        population_structure: Whether to simulate population structure
        n_subpopulations: Number of subpopulations (if population structure is enabled)
        """
        self.n_individuals = n_individuals
        self.n_loci = n_loci
        self.maf_range = maf_range
        self.population_structure = population_structure
        self.n_subpopulations = n_subpopulations
        self.subpop_labels = None
        self.genotype_matrix = None
        self.beta_values = None
        self.fixed_snp = None  # For the single SNP used as a fixed effect
        self.fixed_snp_beta = None  # Effect size for the fixed SNP

    def simulate_allele_freqs(self):
        """
        Simulates allele frequencies within the specified MAF range for each locus.
        """
        min_maf, max_maf = self.maf_range
        return np.random.uniform(min_maf, max_maf, self.n_loci)

    def simulate_genotype_data(self):
        """
        Simulates genotype data with optional population structure.
        return: Genotype matrix (n_individuals x n_loci)
        """
        if self.population_structure:
            return self._simulate_with_population_structure()
        else:
            return self._simulate_without_population_structure()

    def _simulate_without_population_structure(self):
        """
        Simulates genotype data without population structure.
        return: Genotype matrix (n_individuals x n_loci)
        """
        allele_freqs = self.simulate_allele_freqs()
        G = np.zeros((self.n_individuals, self.n_loci))

        for i in range(self.n_loci):
            p = allele_freqs[i]  # Frequency of the alternative allele p
            G[:, i] = np.random.binomial(2, p, size=self.n_individuals)  # Simulate genotypes (0, 1, or 2)

        self.genotype_matrix = G
        return G

    def _simulate_with_population_structure(self): # FST of 0.01
        """
        Simulates genotype data with population structure, dividing individuals into subpopulations.
        return: Genotype matrix (n_individuals x n_loci)
        """
        # Assign individuals to subpopulations
        self.subpop_labels = np.random.choice(self.n_subpopulations, size=self.n_individuals)

        G = np.zeros((self.n_individuals, self.n_loci))
        allele_freqs_global = self.simulate_allele_freqs()  # Global allele frequencies

        # Adjust allele frequencies for each subpopulation
        for i in range(self.n_loci):
            for subpop in range(self.n_subpopulations):
                # Modify allele frequencies for each subpopulation (slightly different from global)
                allele_freqs_subpop = np.clip(allele_freqs_global[i] + np.random.normal(0, 0.05), 0.05, 0.95)
                # Simulate genotypes for individuals in the current subpopulation
                G[self.subpop_labels == subpop, i] = np.random.binomial(2, allele_freqs_subpop, size=(self.subpop_labels == subpop).sum())

        self.genotype_matrix = G
        return G

    def simulate_fixed_snp(self):
        """
        Simulates a single SNP to be used as a fixed effect.
        return: Genotype array (n_individuals) for the fixed SNP
        """
        maf = np.random.uniform(*self.maf_range)  # Minor allele frequency for the fixed SNP
        self.fixed_snp = np.random.binomial(2, maf, size=self.n_individuals)  # Simulate genotypes (0, 1, or 2)
        return self.fixed_snp


    def generate_beta_values(self, effect_distribution='normal', population_specific=False):
        """
        Generates beta values for each locus, optionally considering population structure.

        effect_distribution: Distribution of the effect sizes ('normal' or 'uniform')
        population_specific: If True, generates different beta values for each subpopulation
        return: Beta values (1D array for global, 2D array if population-specific)
        """
        if effect_distribution == 'normal':
            # Generate effect sizes from a normal distribution (mean = 0, std = 1)
            base_beta_values = np.random.normal(0, 1, self.n_loci)
        elif effect_distribution == 'uniform':
            # Generate effect sizes from a uniform distribution (-1, 1)
            base_beta_values = np.random.uniform(-1, 1, self.n_loci)
        else:
            raise ValueError("Invalid effect_distribution. Use 'normal' or 'uniform'.")

        if population_specific and self.population_structure:
            # Generate different beta values for each subpopulation
            self.beta_values = np.zeros((self.n_subpopulations, self.n_loci))
            for subpop in range(self.n_subpopulations):
                self.beta_values[subpop, :] = base_beta_values + np.random.normal(0, 0.1, self.n_loci)
        else:
            # Global beta values
            self.beta_values = base_beta_values

        return self.beta_values

    def compute_grm(self):
        """
        Computes the Genetic Relationship Matrix (GRM).
        return: Genetic Relationship Matrix (GRM)
        """
        if self.genotype_matrix is None:
            raise ValueError("Genotype matrix is not initialized. Run simulate_genotype_data first.")

        # Center genotype matrix by subtracting the mean allele frequency
        allele_means = np.mean(self.genotype_matrix, axis=0) / 2  # Mean allele frequency per locus
        G_centered = self.genotype_matrix - 2 * allele_means  # Centered genotype matrix

        # Compute the variance for each locus
        allele_variances = 2 * allele_means * (1 - allele_means)

        # Standardize the centered genotype matrix by dividing by the standard deviation
        # Avoid division by zero for variance values close to zero
        allele_variances[allele_variances == 0] = 1e-8  # Prevent division by zero
        G_standardized = G_centered / np.sqrt(allele_variances)

        # Compute the GRM as G_centered * G_centered.T / n_loci
        GRM = np.dot( G_standardized,  G_standardized.T) / self.n_loci

        return GRM

    def compute_principal_components(self, n_components=10, plot=False):
        """
        Computes the principal components (PCs) from the genotype matrix.

        n_components: Number of principal components to compute
        plot: Whether to plot the first two principal components
        return: Matrix of principal components (n_individuals x n_components)
        """
        if self.genotype_matrix is None:
            raise ValueError("Genotype matrix is not initialized. Run simulate_genotype_data first.")

        # Standardize the genotype matrix (mean 0, variance 1)
        G_standardized = (self.genotype_matrix - np.mean(self.genotype_matrix, axis=0)) / np.std(self.genotype_matrix, axis=0)

        # Perform PCA
        pca = PCA(n_components=n_components)
        self.principal_components = pca.fit_transform(G_standardized)

        # Plot if requested
        if plot:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=self.principal_components[:, 0], y=self.principal_components[:, 1], hue=self.subpop_labels, palette='viridis', s=100)
            plt.title('PCA of Simulated Genotype Data')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.grid(True)
            plt.show()

        return self.principal_components



class LongitudinalPhenotypeSimulator:
    def __init__(self, n_individuals, repeat_measures, baseline_mean=1000, baseline_sd=100, noise_sd=50, ar1_rho=0.7):
        """
        Initializes the simulation class with parameters for longitudinal phenotypes.

        n_individuals: Number of individuals
        repeat_measures: List specifying the number of repeated measures for each individual
        baseline_mean: Mean of the baseline brain volume (or initial phenotype value)
        baseline_sd: Standard deviation of the baseline brain volume
        noise_sd: Standard deviation of noise for each measurement (controls phenotype variation)
        ar1_rho: Autoregressive correlation coefficient for AR(1) process
        """
        self.n_individuals = n_individuals
        self.repeat_measures = repeat_measures
        self.baseline_mean = baseline_mean
        self.baseline_sd = baseline_sd
        self.noise_sd = noise_sd
        self.ar1_rho = ar1_rho

        if len(repeat_measures) != n_individuals:
            raise ValueError("The length of repeat_measures must match the number of individuals.")

    def simulate_uncorrelated_phenotypes(self):
        """
        Simulates longitudinal phenotypes that are not correlated over time.
        Each time point is independent of the others.

        return: List of arrays where each array contains the simulated uncorrelated phenotypes for an individual.
        """
        phenotypes_list = []

        for i in range(self.n_individuals):
            n_measures = self.repeat_measures[i]

            # Simulate each time point independently from a normal distribution
            individual_phenotypes = np.random.normal(self.baseline_mean, self.baseline_sd, n_measures)

            # Add some noise to the measurements
            individual_phenotypes += np.random.normal(0, self.noise_sd, n_measures)

            phenotypes_list.append(individual_phenotypes)

        flattened_phenotypes_list = [item for sublist in phenotypes_list for item in sublist]

        return flattened_phenotypes_list

    def simulate_autoregressive_phenotypes(self):
        """
        Simulates longitudinal phenotypes with autoregressive correlation (AR(1)).
        The phenotypes are correlated over time with an AR(1) process.

        return: List of arrays where each array contains the simulated autoregressive phenotypes for an individual.
        """
        phenotypes_list = []

        for i in range(self.n_individuals):
            n_measures = self.repeat_measures[i]

            # Simulate the baseline (first time point)
            individual_phenotypes = np.zeros(n_measures)
            individual_phenotypes[0] = np.random.normal(self.baseline_mean, self.baseline_sd)

            # Simulate the autoregressive process for subsequent time points AR(1)
            for t in range(1, n_measures):
                # Volume at time t depends on the volume at time t-1 plus random noise
                individual_phenotypes[t] = self.ar1_rho * individual_phenotypes[t-1] + np.random.normal(0, self.noise_sd)

            phenotypes_list.append(individual_phenotypes)

        flattened_phenotypes_list = [item for sublist in phenotypes_list for item in sublist]

        return flattened_phenotypes_list




class CovarianceSimulator:
    def __init__(self, n_individuals, repeat_measures):
        """
        Initializes the simulator with parameters for longitudinal data simulation.

        n_individuals: Number of individuals
        repeat_measures: Array containing the number of measurements for each individual
        """
        self.n_individuals = n_individuals
        self.repeat_measures = repeat_measures

    def generate_covariance_matrix(self, num_measures, rho=0.5):
        """
        Generate a covariance matrix with specified correlation (rho) between successive measurements.

        num_measures: Number of measurements for the covariance matrix
        rho: Correlation coefficient between successive measurements
        return: A covariance matrix
        """
        # Constructing an autoregressive covariance matrix
        cov_matrix = rho ** np.abs(np.subtract.outer(np.arange(num_measures), np.arange(num_measures)))
        return cov_matrix

    def simulate_longitudinal_data(self):
        """
        Simulates longitudinal data for a set number of subjects, each with a specified number of measurements.
        Includes dynamic covariates (Age and ICV) and a static covariate (Sex).

        return: A pandas DataFrame with simulated data
        """
        data = []

        # Static measurements (these values are generated once per individual)
        prob_female = 0.3
        sexes = np.random.binomial(1, prob_female, size=self.n_individuals)  # 1 = female, 0 = male

        for i, num_measures in enumerate(self.repeat_measures):
            age_cov_matrix = self.generate_covariance_matrix(num_measures, rho=0.8)
            icv_cov_matrix = self.generate_covariance_matrix(num_measures, rho=0.5)

            # Simulate baseline and increments for age and ICV
            initial_age = np.random.randint(20, 60)
            initial_icv = np.random.normal(1400, 100)
            age_increments = multivariate_normal.rvs(mean=np.linspace(0, num_measures - 1, num_measures), cov=age_cov_matrix)
            icv_values = multivariate_normal.rvs(mean=np.full(num_measures, initial_icv), cov=icv_cov_matrix)

            ages = initial_age + age_increments
            icvs = icv_values

            # Generate measurements with no specified correlation (for simplicity)
            measurements = np.random.normal(100 - 0.5 * ages, 5, num_measures)  # Example measurement depending on age

            for time_point in range(num_measures):
                data.append([
                    i,  # Subject ID
                    time_point,  # Time Point
                    sexes[i],  # Static: Sex
                    ages[time_point],  # Dynamic: Age
                    icvs[time_point],  # Dynamic: ICV
                ])

        df = pd.DataFrame(data, columns=['SubjectID', 'TimePoint', 'Sex', 'Age', 'ICV'])
        return df

def run_simulation(n_individuals, n_loci, maf_range, repeat_measures, baseline_mean, baseline_sd, noise_sd, ar1_rho, covar_dynamic_name_list, n_simulations):
    simulation_results = []
    times = []
    peak_memory_usage = []
    total_time = 0  # To track total time across all simulations

    # Loop through each simulation
    for i in range(n_simulations):
        start_time = time.time()  # Record the start time of the simulation

        # Start tracking memory allocations
        tracemalloc.start()

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
        model.fit(autoregressive_phenotypes, covariance_simulator_data, grm, repeat_measures, subject_level=None)


        # Store the results in the results list
        model.var_c_est = model.var_c_est if hasattr(model, 'var_c_est') else 'N/A'
        simulation_results.append([model.rho_est, model.var_g_est, model.var_t_est, model.var_c_est, model.var_e_est, model.h2_est, model.beta_est])

        # Calculate the elapsed time for this simulation
        end_time = time.time()
        elapsed_time = end_time - start_time
        #total_time += elapsed_time  # Add to the total time
        times.append(elapsed_time)

        # Stop tracking memory allocations
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_memory_usage.append(peak / 10**6)  # Convert peak memory usage to MB

    # Calculate the average time for all simulations
    # average_time = total_time / n_simulations
    # return average_time, simulation_results
    return np.mean(times), np.mean(peak_memory_usage), simulation_results


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
    average_times = []

    for n_individuals in subject_numbers:
        repeat_measures = np.repeat(3, n_individuals)  # Repeat measures for each individual
        #print(f"Running simulation for {n_individuals} individuals...")
        print("Running simulation for {} individuals...".format(n_individuals))
        # Run the simulation and get the average running time
        average_time, _ = run_simulation(n_individuals, n_loci, maf_range, repeat_measures, baseline_mean, baseline_sd, noise_sd, ar1_rho, covar_dynamic_name_list, n_simulations)

        # Append the average time for this number of individuals
        average_times.append(average_time)
        #print(f"Average time for {n_individuals} individuals: {average_time:.2f} seconds")
        print("Average time is {} seconds...".format(average_time))
    # Plot the average running time against the number of subjects using a dot plot and lines
    plt.figure(figsize=(8, 6))
    plt.scatter(subject_numbers, average_times, color='b', s=100, label='Average Running Time')  # Scatter plot (dot plot)
    plt.plot(subject_numbers, average_times, linestyle='-', color='b', label='Trend Line')  # Line plot to connect the dots
    plt.title('Average Running Time vs Number of Subjects')
    plt.xlabel('Number of Subjects')
    plt.ylabel('Average Running Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


