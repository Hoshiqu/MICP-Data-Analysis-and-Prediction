Project Overview
This project aims to develop and evaluate machine learning models to predict mercury injection capillary pressure (MICP) experiment results from core sample data obtained from different wells. The dataset contains various features related to core samples, including porosity, permeability, density, lithology, gamma ray logs, and computer tomographic measurements. The goal is to build a predictive model that accurately estimates the MICP outputs ('bv_0' to 'bv_100' and 'pc_0' to 'pc_100') based on the available features.
Tasks:
1. Baseline Model Creation: Develop a simple baseline model for comparison purposes, such as a naive predictor.
2. Model Development: Construct a machine learning model to predict the MICP outputs using the provided dataset.
3. Data Preprocessing: Apply appropriate techniques for data preprocessing, including normalization, feature selection, and feature engineering, to enhance model performance.
4. Model Evaluation: Evaluate the performance of the developed model using relevant metrics and a suitable cross-validation strategy. Utilize the test data for final assessment.
5. Hyperparameter Tuning: Fine-tune the model's hyperparameters to optimize its performance if necessary.
6. Baseline Comparison: Conduct statistical analysis to demonstrate that the developed model outperforms the baseline model.
7. Results Visualization: Visualize the predictions of the model to gain insights into the data and model performance.
8. Model Improvement Suggestions: Provide recommendations for improving the model based on insights gained and areas identified for enhancement.
9. Summary: Summarize the model development process, key findings, and results in English.
Dataset Description:
The dataset comprises core sample data from different wells, with features including depth, porosity, permeability, density, lithology, gamma ray logs, and computer tomographic measurements. These features serve as inputs for predicting the MICP outputs.


python.exe -m pip install --upgrade pip

pip install -r requirements.txt.

streamlit run test.py


In the initial part of the test.py program, constants are declared that can be modified 
according to your preferences. I opted to utilize randomization, but in the event 
that you require replicating a specific seed, I recommended the appropriate tools.
