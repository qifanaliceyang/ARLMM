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



class LinearModel:
    """
    A class to define and fit a linear model (no grouping variables).
    """
    def __init__(self, formula, data):
        """
        Initialize the linear model.
        :param formula: str, the formula for the linear model (e.g., "y ~ x1 + x2").
        :param data: DataFrame, the dataset to fit the model.
        """
        self.formula = formula
        self.data = data
        self.model = None
        self.result = None

    def fit(self):
        """
        Fit the linear model using statsmodels.
        """
        self.model = smf.ols(self.formula, self.data)
        self.result = self.model.fit()
        return self.result

    def get_p_value(self, variable):
        """
        Retrieve the p-value for a specified variable.
        :param variable: str, the name of the fixed effect variable to check.
        :return: float, the p-value of the specified variable.
        """
        if self.result is None:
            raise ValueError("Model is not fitted. Call the fit() method first.")
        return self.result.pvalues[variable]


class PowerAndTypeIErrorCalculator:
    """
    A class to calculate statistical power and Type I error rate for a linear model.
    """
    def __init__(self, lm, effect_var, n_simulations=1000, alpha=0.05):
        """
        Initialize the calculator.
        :param lm: LinearModel, the model for which to calculate power and Type I error rate.
        :param effect_var: str, the name of the variable whose effect size is being tested.
        :param n_simulations: int, the number of simulations to run (default: 1000).
        :param alpha: float, the significance level (default: 0.05).
        """
        self.lm = lm
        self.effect_var = effect_var
        self.n_simulations = n_simulations
        self.alpha = alpha

    def simulate_data(self, true_effect, random_state=None):
        """
        Simulate data with a specified true effect size.
        :param true_effect: float, the true effect size for the variable.
        :param random_state: int, random seed for reproducibility.
        :return: DataFrame, the simulated dataset.
        """
        if random_state:
            np.random.seed(random_state)

        n_obs = len(self.lm.data)

        # Simulate independent variable
        x = self.lm.data[self.effect_var]

        # Simulate dependent variable
        y = true_effect * x + np.random.normal(0, 1, n_obs)

        # Return simulated data
        sim_data = self.lm.data.copy()
        sim_data['y'] = y
        return sim_data

    def calculate_power(self, true_effect):
        """
        Calculate power based on simulations.
        :param true_effect: float, the true effect size for the variable.
        :return: float, the estimated power.
        """
        significant_count = 0

        for i in range(self.n_simulations):
            sim_data = self.simulate_data(true_effect=true_effect, random_state=i)
            self.lm.data = sim_data
            result = self.lm.fit()

            if result.pvalues[self.effect_var] < self.alpha:
                significant_count += 1

        power = significant_count / self.n_simulations
        return power

    def calculate_type_I_error(self):
        """
        Calculate the Type I error rate under the null hypothesis (true effect = 0).
        :return: float, the Type I error rate.
        """
        return self.calculate_power(true_effect=0)


# Example Usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        "x1": np.random.normal(size=100)
    })
    data["y"] = 0.5 * data["x1"] + np.random.normal(size=100)

    # Define and fit the linear model
    lm = LinearModel("y ~ x1", data)

    # Power and Type I error calculations
    calculator = PowerAndTypeIErrorCalculator(lm, effect_var="x1", n_simulations=100, alpha=0.05)

    # Calculate power for a true effect size of 0.5
    power = calculator.calculate_power(true_effect=0.5)
    print(f"Estimated Power: {power}")

    # Calculate Type I error rate (true effect size = 0)
    type_I_error = calculator.calculate_type_I_error()
    print(f"Estimated Type I Error Rate: {type_I_error}")


if __name__ == "__main__":
    main()


