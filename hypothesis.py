from statsmodels.multivariate.manova import MANOVA
import pandas as pd
import pingouin as pg
from loader import load_data
import numpy as np
from enums import Algorithm, get_algorithm_name
from loader import dependent_variables
from scipy.stats import f_oneway, levene, ttest_ind, shapiro, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def normality_test(data: np.ndarray, alpha=0.05):
    """
    Perform normality test on data.
    """
    # Perform Shapiro-Wilk test
    stat, p_value = shapiro(data)

    print(f"Shapiro-Wilk Test Statistic: {stat}")
    print(f"P-value: {p_value}")

    # Check for normality based on the p-value
    alpha = 0.05
    if p_value >= alpha:
        print("Data looks normally distributed (fail to reject H0)")
        return True
    else:
        print("Data does not look normally distributed (reject H0)")
        return False

def homoscedasticity_test(alpha: float, *groups):
    """
    Perform variance test for t_test
    """
    statistic, p_value = levene(*groups)

    # Print the results
    print(f"Levene's Test Statistic: {statistic}")
    print(f"P-value: {p_value}")

    if p_value < alpha:
        print("Result: Reject the null hypothesis. There is evidence of unequal variances among groups.")
        return False
    else:
        print("Result: Fail to reject the null hypothesis. There is not enough evidence of unequal variances among groups.")
        return True


def anova(alpha: float, *groups: np.ndarray):
    homoscedasticity_test(alpha, *groups)
    
    # Perform one-way ANOVA
    statistic, p_value = f_oneway(*groups)

    # Print the results
    print("ANOVA Statistic:", statistic)
    print("P-value:", p_value)

    # Interpret the results
    if p_value < alpha:
        print("Reject the null hypothesis.")
    else:
        print("Fail to reject the null hypothesis. No significant differences among group means.")

def t_test(a: np.ndarray, b: np.ndarray, alternative: str, alpha=0.05):
    """
    Perform test to verify whether variable is less/greater/not equal.
    Based on properties of data such distribution and variance, the most
    fitting test will be chosen. 
    """

    normality1, normality2 = normality_test(a, alpha), normality_test(b, alpha)

    if normality1 and normality2:
        # Perform independent two-sample t-test
        equal_variance = homoscedasticity_test(alpha, a, b)
        print(f"Performing {'Student test' if equal_variance else 'Welch'}")
        statistic, p_value = ttest_ind(a, b, alternative=alternative, equal_var=equal_variance)
    else:
        print(f"Performing Mannwhitneyu")
        statistic, p_value = mannwhitneyu(a, b, alternative=alternative)


    # Print the results
    print("T-test Statistic:", statistic)
    # print("Df: ", df)
    print("P-value:", p_value)

    # Interpret the results
    if p_value < alpha:
        print(f"Reject the null hypothesis. It is {alternative or 'unequal'}")
    else:
        print("Fail to reject the null hypothesis. No significant difference between group means.")

# obsolete
def hypothesis_1(data: str):
    df = load_data(data)
    df = df.loc[df["Algorithm"].isin([get_algorithm_name(Algorithm.Random), get_algorithm_name(Algorithm.SemiRandom)]), ["Algorithm", "Units traveled"]]
    rand = df.loc[df["Algorithm"] == get_algorithm_name(Algorithm.Random), "Units traveled"]
    semi_random = df.loc[df["Algorithm"] == get_algorithm_name(Algorithm.SemiRandom), "Units traveled"]
    
    t_test(semi_random, rand, alternative="less")

# MANOVA attempt. Not used, although it verified, that A* Sequential and A* Orientation
# are closely related.
def hypothesis_2(data: str):
    alpha = 0.05
    df = load_data(data)
    # algo_names = [algo.name for algo in algorithms]
    # anova(*[df.loc[df["Algorithm"] == algo] for algo in algo_names])

    variables = [f"Q('{var}')" for var in dependent_variables]
    manova_model = MANOVA.from_formula(f"{' + '.join(variables)} ~ Algorithm", data=df)
    manova_results = manova_model.mv_test()
    print(manova_results)
    
    homoscedasticity = homoscedasticity_test(alpha, *[df[var] for var in dependent_variables])

    if homoscedasticity:
        # Perform Tukey's HSD post-hoc test
        tukey_results = pairwise_tukeyhsd(df["Units traveled"], df["Algorithm"])
    else:
        for var in dependent_variables:
            print(f"Data form {var}")
            result = pg.pairwise_gameshowell(data=df, dv=var, between="Algorithm")
            print(result)

# obsolete
def hypothesis_3(data: str):
    df = load_data(data)
    df = df.loc[df["Algorithm"].isin([get_algorithm_name(Algorithm.AStarOrientation), get_algorithm_name(Algorithm.AStarSequential)]), ["Algorithm", "Rotation accumulator"]]
    orientation = df.loc[df["Algorithm"] == get_algorithm_name(Algorithm.AStarOrientation), "Rotation accumulator"]
    sequential = df.loc[df["Algorithm"] == get_algorithm_name(Algorithm.AStarSequential), "Rotation accumulator"]
    
    t_test(orientation, sequential, alternative="less")

# obsolete
def hypothesis_4(data: str):
    df = load_data(data)
    df = df.loc[df["Algorithm"].isin([get_algorithm_name(Algorithm.AStarOrientation), get_algorithm_name(Algorithm.AStarSequential)]), ["Algorithm", "Error"]]
    orientation = df.loc[df["Algorithm"] == get_algorithm_name(Algorithm.AStarOrientation), "Error"]
    sequential = df.loc[df["Algorithm"] == get_algorithm_name(Algorithm.AStarSequential), "Error"]
    
    t_test(orientation, sequential, alternative="less")

# obsolete
def hypothesis_5(data: str):
    df = load_data(data)
    df = df.loc[df["Algorithm"].isin([get_algorithm_name(Algorithm.AStarOrientation), get_algorithm_name(Algorithm.AStarSequential)]), ["Algorithm", "Time taken"]]
    orientation = df.loc[df["Algorithm"] == get_algorithm_name(Algorithm.AStarOrientation), "Time taken"]
    sequential = df.loc[df["Algorithm"] == get_algorithm_name(Algorithm.AStarSequential), "Time taken"]
    
    t_test(orientation, sequential, alternative="less")

def hypothesis_custom(data: str, algo1: str, algo2: str, variable: str, alternative: str):
    """
    Compare 2 data arrays according to alternate hypothesis using Student T-test,
    Manwhitney U Test or Welch test.
    """
    df = load_data(data)
    df = df.loc[df["Algorithm"].isin([algo1, algo2]), ["Algorithm", variable]]
    orientation = df.loc[df["Algorithm"] == algo1, variable]
    sequential = df.loc[df["Algorithm"] == algo2, variable]
    
    t_test(orientation, sequential, alternative=alternative)