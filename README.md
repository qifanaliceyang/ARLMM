# Autoregressive Linear Mixed Models 
[![Contributors](https://img.shields.io/github/contributors/qifanaliceyang/ARLMM)](https://github.com/qifanaliceyang/ARLMM/graphs/contributors) 
[![Issues](https://img.shields.io/github/issues/qifanaliceyang/ARLMM)](https://github.com/qifanaliceyang/ARLMM/issues)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
## 🚀 Overview


Autoregressive Linear Mixed Models (ARLMM) is a fast, accurate, sensitivity and specificity balanced mixed model approach to analyze genetic associations with temporally or spatially correlated phenotypes. It provides dynamic profiles of genetic effects on human brain changes during disease progression. 


## 🎯 Features
- **Scalability**: Running large-scale genetic association experiments for general populations (N >= 10, 000) and handling repeated measurements (t >= 3) efficiently across multiple GPUs. 
- **Predictability**: Using autoregressive models to improve predictive abilities for unseen subjects' brain imaging measurements, and future brain imaging measurements of observed subjects (in progress).
- **Robustness**: Incorporating support vector regression to model covariance structures of repeated measurements.
- **Power**: Achieving higher power for small clinical samples (N < 1, 000) in case-control studies. 

## 🛠️ Installation
Our package will result in an error if used in any Python version prior to 3.6. The following dependencies must be installed: NumPy, pandas, SciPy, scikit-learn, and Matplotlib.
1. Clone or download the repository:
   ```bash
   git clone https://github.com/qifanaliceyang/ARLMM
2. Creat a data directory and upload your data:
   ```bash
   mkdir -p data
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## 🧑‍💻 Code Examples
How to run the permutation tests:
   ```bash
   python permutation_test.py --help
   ```
```
Arguments:

--n_loci: int, default 1500
		Number of loci (genetic markers).
  
--n_simulations: int, default 5
		Number of simulations to run.
  
--n_parameters: int, default 7
		Number of parameters to estimate.
  
--maf_range_low: float, default 0.05
		Lower bound of the MAF range.
  
--maf_range_high: float, default 0.4
		Upper bound of the MAF range.
  
--baseline_mean: float, default 3000
		Mean baseline phenotype means.
  
--baseline_sd: float, default 100
		Standard deviation of the baseline phenotypes.
  
--noise_sd: float, default 50
		Standard deviation of random noise with respect to baseline phenotypic measurements.
  
--ar1_rho: float, default 0.8
		AR(1) Autoregressive parameter for simulating correlated phenotypes.
  
--covars: list of str, default ['Age', 'Sex', 'ICV']
		Covariate names. Provide multiple values separated by spaces.
  
--subject_numbers: list of int, default [100]
		Number of subjects to run simulations. You can provide multiple values to run multiple simulations.
```
Examples:
1. Run with a larger number of loci and two different subject sizes:
   ```bash
   python permutation_test.py --n_loci 10000 --subject_numbers 500 1000
2. Change the MAF range and run more simulations:
   ```bash
   python permutation_test.py --maf_range_low 0.1 --maf_range_high 0.35 --n_simulations 1000
3. Specify covariates and increase standard deviation of noises:
   ```bash
   python permutation_test.py --covars Age Sex ICV --noise_sd 75


## 👥 Authors
Thanks for the contributions and collaborations from 
- **Imaging Genetics Center** - https://github.com/USC-IGC
- **ENIGMA Consortium** - https://github.com/ENIGMA-git

Please contact **Qifan Yang** (qifan.yang@usc.edu) if you have any comments, suggestions or questions.

## 🌟 Acknowledgements
This package is developed using the UK Biobank dataset (https://www.ukbiobank.ac.uk) under Application Number 11559, and datasets from Alzheimer's Disease Neuroimaging Initiative (ADNI) (https://adni.loni.usc.edu) with NIH grant R01-AG059874 (PI: Jahanshad).

## 📚 Citations
Please cite ARLMM version 1.0 publised in 2019, and we are working on publishing the version 2.0 soon.


Yang, Qifan, et al. "Support vector based autoregressive mixed models of longitudinal brain changes and corresponding genetics in Alzheimer’s disease." Predictive Intelligence in Medicine: Second International Workshop, PRIME 2019, Held in Conjunction with MICCAI 2019, Shenzhen, China, October 13, 2019, Proceedings 2. Springer International Publishing, 2019.
