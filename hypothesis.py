import statsmodels
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, levene, ttest_ind

def anova(*groups: np.ndarray):
    # Perform Levene's test
    statistic, p_value = levene(*groups)

    # Print the results
    print(f"Levene's Test Statistic: {statistic}")
    print(f"P-value: {p_value}")

    # Check the homoscedasticity based on the p-value
    if p_value > 0.05:
        print("The variances are homogeneous (homoscedastic).")
    else:
        print("The variances are not homogeneous (heteroscedastic).")
        raise ValueError()
    
    # Perform one-way ANOVA
    statistic, p_value = f_oneway(*groups)

    # Print the results
    print("ANOVA Statistic:", statistic)
    print("P-value:", p_value)

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis. There are significant differences among group means.")
    else:
        print("Fail to reject the null hypothesis. No significant differences among group means.")

def t_test(a: np.ndarray, b: np.ndarray):
    # Perform independent two-sample t-test
    statistic, p_value = ttest_ind(a, b)

    # Print the results
    print("T-test Statistic:", statistic)
    print("P-value:", p_value)

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference between group means.")
    else:
        print("Fail to reject the null hypothesis. No significant difference between group means.")
