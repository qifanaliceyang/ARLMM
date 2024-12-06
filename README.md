# Autoregressive Linear Mixed Models 
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
