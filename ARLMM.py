import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from scipy.linalg import block_diag, inv
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.stats import t, zscore

class ARLMM_Model:
    def __init__(self):
        pass

    def mapping_matrix_T(self, measurement):
        """
        Get mapping matrix to map cross-sectional variables to longitudinal information.
        """
        T = np.expand_dims(np.repeat(1, measurement[0]), axis=1)
        for i in range(1, len(measurement)):
            T = block_diag(T, np.expand_dims(np.repeat(1, measurement[i]), axis=1))
        return T

    def cross_sectional_longit_mapping_function(self, matrix_T, correlation_matrix):
        """
        Map cross-sectional covariate information or random effect correlation information to longitudinal.
        """
        T_correlation = np.matmul(matrix_T, correlation_matrix)
        T_correlation_T_transpose = np.matmul(T_correlation, np.transpose(matrix_T))
        longit_correlation = T_correlation_T_transpose
        return longit_correlation

    def load_groups_from_file(self, measurement):
        """
        Create group information for cross-validation based on measurements.
        """
        k = 0
        groups = []
        for cnt_meas in measurement:
            groups.extend([k] * cnt_meas)
            k += 1
        groups = np.array(groups)
        print(groups)
        return groups

    def load_random_effect_info_from_file(self, grm, measurement, subject_level, T):
        """
        Get the random effect covariance matrices.
        """
        # Create longitudinal GRM
        grm_arr = np.array(grm)
        longit_grm = self.cross_sectional_longit_mapping_function(T, grm_arr)

        # Create measurement error info
        longit_error = np.identity(np.sum(measurement))

        # Create temporal or spatially correlated random effect matrices
        if measurement is not None:
            # Initialize C1 and C2
            longit_C1 = np.zeros((measurement[0], measurement[0]))
            longit_C2 = np.zeros((measurement[0], measurement[0]))
            for i in range(measurement[0]):
                for j in range(measurement[0]):
                    if abs(i - j) == 1:
                        longit_C1[i, j] = 1
                    if abs(i - j) == 2:
                        longit_C2[i, j] = 1

            for s in range(1, len(measurement)):
                longit_C1_temp = np.zeros((measurement[s], measurement[s]))
                longit_C2_temp = np.zeros((measurement[s], measurement[s]))
                for i in range(measurement[s]):
                    for j in range(measurement[s]):
                        if abs(i - j) == 1:
                            longit_C1_temp[i, j] = 1
                        if abs(i - j) == 2:
                            longit_C2_temp[i, j] = 1
                longit_C1 = block_diag(longit_C1, longit_C1_temp)
                longit_C2 = block_diag(longit_C2, longit_C2_temp)

            # Create site effect matrix if subject_level is not None
            if subject_level is not None:
                subject_level_dummy = pd.get_dummies(subject_level)
                subject_level_cnt = subject_level_dummy.sum(axis=0)
                block_diag_matrix = np.identity(subject_level_cnt.iloc[0])
                for i in range(1, len(subject_level_cnt)):
                    block_diag_matrix = block_diag(block_diag_matrix, np.identity(subject_level_cnt.iloc[i]))
                longit_block_diag_matrix = self.cross_sectional_longit_mapping_function(T, block_diag_matrix)
                return longit_grm, longit_C1, longit_C2, longit_block_diag_matrix, longit_error
            else:
                return longit_grm, longit_C1, longit_C2, longit_error
        else:
            return longit_grm, longit_error

    def safe_inverse(self, X, lambda_value=0.01):
      XX = np.dot(X.T, X)
    
      try:
          # Attempt to compute the inverse of XX
          XX_inv = inv(XX)
          #print("Matrix is invertible, using regular inverse.")
      
      except np.linalg.LinAlgError as err:
          # If non-invertible, add lambda_value to the diagonal and compute the regularized inverse
          print("Matrix is non-invertible, using regularized inverse with lambda =", lambda_value)
          I = np.identity(XX.shape[0])
          XX_inv = inv(XX + lambda_value * I)
    
      return XX_inv

    def project_covariate_func(self, X):
        """
        Get the projection matrix for covariates.
        """
        XX = np.dot(np.transpose(X), X)
        Z = np.dot(X, self.safe_inverse(XX))
        P = np.identity(X.shape[0]) - np.dot(Z, np.transpose(X))
        return P

    def fit(self, Y, X, grm, measurement, subject_level=None):
        """
        Fit the ARLMM model.
        """
        T = self.mapping_matrix_T(measurement)
        P = self.project_covariate_func(X)
        Y_Y_transpose_1d = np.reshape(np.dot(Y, np.transpose(Y)), [-1, 1], order='F')

        if subject_level is not None:
            longit_grm, longit_C1, longit_C2, longit_block_diag_matrix, longit_error = self.load_random_effect_info_from_file(grm, measurement, subject_level, T)
            project_longit_grm_1d = np.reshape(self.cross_sectional_longit_mapping_function(P, longit_grm), [-1, 1], order='F')
            project_longit_C1_1d = np.reshape(self.cross_sectional_longit_mapping_function(P, longit_C1), [-1, 1], order='F')
            project_longit_C2_1d = np.reshape(self.cross_sectional_longit_mapping_function(P, longit_C2), [-1, 1], order='F')
            project_longit_block_diag_matrix_1d = np.reshape(self.cross_sectional_longit_mapping_function(P, longit_block_diag_matrix), [-1, 1], order='F')
            project_longit_error_1d = np.reshape(self.cross_sectional_longit_mapping_function(P, longit_error), [-1, 1], order='F')

            project_longit_X_1d = np.concatenate((project_longit_grm_1d, project_longit_C1_1d, project_longit_C2_1d, project_longit_block_diag_matrix_1d, project_longit_error_1d), axis=1)

            clf = SGDRegressor(tol=1e-3, penalty='l2', loss='squared_epsilon_insensitive', fit_intercept=False, early_stopping=True)
            clf.fit(project_longit_X_1d, Y_Y_transpose_1d.ravel())

            sigma = clf.coef_
            rho_est = max(1e-11, min(sigma[2]/sigma[1], 1))
            var_t_est = max(0.5 * sigma[1] / rho_est + 0.5 * sigma[2] / (rho_est ** 2), 1e-11)
            var_g_est = max(sigma[0], 1e-11)
            var_c_est = max(sigma[3], 1e-11)
            var_e_est = max(sigma[4] - var_t_est, 1e-11)
            h2_est = var_g_est / (var_t_est + var_g_est + var_c_est + var_e_est)

            t_error_cov = var_t_est * (np.identity(T.shape[0]) + rho_est * longit_C1 + rho_est ** 2 * longit_C2)
            error_cov = var_e_est * np.identity(T.shape[0])
            genetic_cov = var_g_est * longit_grm
            correlation_cov = var_c_est * longit_block_diag_matrix
            total_V = t_error_cov + error_cov + genetic_cov + correlation_cov

            Y_new = np.dot(inv(total_V), Y)
            response_new = np.dot(np.transpose(X), Y_new)
            X_new = np.dot(inv(total_V), X)
            predictor_new = np.dot(np.transpose(X), X_new)
            beta_est = np.dot(inv(predictor_new), response_new)

            self.rho_est = rho_est
            self.var_g_est = var_g_est
            self.var_t_est = var_t_est
            self.var_c_est = var_c_est
            self.var_e_est = var_e_est
            self.h2_est = h2_est
            self.beta_est = beta_est

        else:
            longit_grm, longit_C1, longit_C2, longit_error = self.load_random_effect_info_from_file(grm, measurement, subject_level, T)
            project_longit_grm_1d = np.reshape(self.cross_sectional_longit_mapping_function(P, longit_grm), [-1, 1], order='F')
            project_longit_C1_1d = np.reshape(self.cross_sectional_longit_mapping_function(P, longit_C1), [-1, 1], order='F')
            project_longit_C2_1d = np.reshape(self.cross_sectional_longit_mapping_function(P, longit_C2), [-1, 1], order='F')
            project_longit_error_1d = np.reshape(self.cross_sectional_longit_mapping_function(P, longit_error), [-1, 1], order='F')

            project_longit_X_1d = np.concatenate((project_longit_grm_1d, project_longit_C1_1d, project_longit_C2_1d, project_longit_error_1d), axis=1)

            clf = SGDRegressor(tol=1e-3, penalty='l2', loss='squared_epsilon_insensitive', fit_intercept=False, early_stopping=True)
            clf.fit(project_longit_X_1d, Y_Y_transpose_1d.ravel())

            sigma = clf.coef_
            rho_est = max(1e-11, min(sigma[2]/sigma[1], 1))
            var_t_est = max(0.5 * sigma[1] / rho_est + 0.5 * sigma[2] / (rho_est ** 2), 1e-11)
            var_g_est = max(sigma[0], 1e-11)
            var_e_est = max(sigma[3] - var_t_est, 1e-11)
            h2_est = var_g_est / (var_t_est + var_g_est + var_e_est)

            t_error_cov = var_t_est * (np.identity(T.shape[0]) + rho_est * longit_C1 + rho_est ** 2 * longit_C2)
            error_cov = var_e_est * np.identity(T.shape[0])
            genetic_cov = var_g_est * longit_grm
            total_V = t_error_cov + error_cov + genetic_cov

            Y_new = np.dot(inv(total_V), Y)
            response_new = np.dot(np.transpose(X), Y_new)
            X_new = np.dot(inv(total_V), X)
            predictor_new = np.dot(np.transpose(X), X_new)
            beta_est = np.dot(inv(predictor_new), response_new)

            self.rho_est = rho_est
            self.var_g_est = var_g_est
            self.var_t_est = var_t_est
            self.var_e_est = var_e_est
            self.h2_est = h2_est
            self.beta_est = beta_est

        return self  # For method chaining

def main():
    # Example data (Replace with actual data loading)
    np.random.seed(0)  # For reproducibility

    # Dummy response variable Y
    Y = np.random.randn(100)
    Y = np.expand_dims(Y, axis=1) 

    # Dummy design matrix X with intercept and 4 covariates
    X = np.random.randn(100, 5)
    X[:, 0] = 1  # Intercept

    # Dummy GRM for 20 subjects
    grm = np.random.randn(20, 20)
    grm = (grm + grm.T) / 2  # Make it symmetric

    # Each subject has 5 measurements
    measurement = [5] * 20

    # Dummy subject levels (could be None)
    subject_level = np.random.choice(['A', 'B', 'C'], size=20)

    # Initialize and fit the model
    model = ARLMM_Model()
    model.fit(Y, X, grm, measurement, subject_level)

    # Access and print the estimated parameters
    print('rho_est:', model.rho_est)
    print('var_g_est:', model.var_g_est)
    print('var_t_est:', model.var_t_est)
    print('var_c_est:', model.var_c_est if hasattr(model, 'var_c_est') else 'N/A')
    print('var_e_est:', model.var_e_est)
    print('h2_est:', model.h2_est)
    print('beta_est:', model.beta_est)

if __name__ == '__main__':
    main()
