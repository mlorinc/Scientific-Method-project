import seaborn as sns
import pandas as pd
import re
from sklearn.decomposition import PCA
from enums import Algorithm, get_algorithm_name
from matplotlib import pyplot as plt
from pathlib import Path
from loader import dependent_variables, load_data, get_graph_ylabel

# Specify order of algorithms on X axis
order = [
    get_algorithm_name(Algorithm.Random),
    get_algorithm_name(Algorithm.SemiRandom),
    get_algorithm_name(Algorithm.AStarRandom),
    get_algorithm_name(Algorithm.AStarSequential),
    get_algorithm_name(Algorithm.AStarOrientation)
]

def setup(output: str, format: str):
    """
    Perform file format check and folder existence check.
    """
    pattern = r"^[a-zA-Z0-9-]+$"
    
    # Use re.match to check if the extension matches the pattern
    if not re.match(pattern, format):
        raise ValueError(f"invalid given format: {format}")

    output_path: Path = Path(output) if output else Path("figures")
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path, format

def generate_boxplot(data: str, output: str, format="png"):
    """
    Generate boxplot for multiple dependent variables, which are
    sorted based on Map and Algorithm.
    """
    output_path, format = setup(output, format)
    # Set figure size to A4 landscape (in inches)
    # a4_width_inch = 11.69 // 2  # 29.7 cm
    # a4_height_inch = 8.27  # 21.0 cm

    # plt.figure(figsize=(a4_width_inch, a4_height_inch))

    df = load_data(data)
    for independent_var in ["Algorithm"]:
        for dependent_var in dependent_variables:
            ax = sns.boxplot(data=df, y=dependent_var, x=independent_var, hue="Map", order=order)
            ax.set(yscale="log", ylabel=get_graph_ylabel(dependent_var))
            plt.savefig(output_path / f"{independent_var}_{dependent_var}.{format}".lower().replace(" ", "_"))
            plt.clf()

def generate_hypothesis_1_boxplot(data: str, output: str, format="png"):
    """
    Generate boxplots for stochastic algorithms.
    """
    output_path, format = setup(output, format)
    df = load_data(data)
    df = df.loc[df["Algorithm"].isin([get_algorithm_name(Algorithm.Random), get_algorithm_name(Algorithm.SemiRandom)]), :]
    print(df)
    for dependent_var in dependent_variables:
        ax = sns.boxplot(data=df, y=dependent_var, x="Algorithm", hue="Map", order=order)
        ax.set(yscale="log", ylabel=get_graph_ylabel(dependent_var))
        plt.savefig(output_path / f"stochastic_boxplot_{dependent_var}.{format}".lower().replace(" ", "_"))
        plt.clf()


def generate_hypothesis_2_boxplot(data, output, format):
    """
    Generate boxplots, although do not take map into account.
    """
    output_path, format = setup(output, format)
    df = load_data(data)

    for var in dependent_variables:
        df[dependent_variables] = df[dependent_variables] / df[dependent_variables].max()

    ax = None
    for var in dependent_variables:
        ax = sns.boxplot(data=df, y=var, x="Algorithm", order=order)
    ax.set(ylabel="Time taken (normalized)")
    plt.savefig(output_path / f"perfomance.{format}".lower().replace(" ", "_"))
    plt.clf()

    # Apply PCA to reduce from 4D to 2D
    # pca = PCA(n_components=2)
    # print(df[dependent_variables].to_numpy())
    # X = pca.fit_transform(df[dependent_variables].to_numpy())

    # # Create a DataFrame for the reduced 2D data
    # df_reduced = pd.DataFrame(X, columns=['X', 'Y'])

    # df_reduced["Algorithm"] = df["Algorithm"].to_numpy()
    # df_reduced["Map"] = df["Map"].to_numpy()
    
    # ax = sns.scatterplot(data=df_reduced, x="X", y="Y", hue="Algorithm")
    # ax.set(yscale="log")
    # plt.savefig(output_path / f"perfomance.{format}".lower().replace(" ", "_"))
    # plt.clf()

def generate_astar_boxplot(data: str, output: str, format="png"):
    """
    Generate A* specific box plots.
    """
    output_path, format = setup(output, format)
    df = load_data(data)
    df = df.loc[df["Algorithm"].isin([get_algorithm_name(Algorithm.AStarOrientation), get_algorithm_name(Algorithm.AStarSequential)]), :]
    # print(df)
    for dependent_var in dependent_variables:
        ax = sns.boxplot(data=df, y=dependent_var, x="Algorithm", hue="Map", order=order)
        ax.set(ylabel=get_graph_ylabel(dependent_var))
        plt.savefig(output_path / f"astar_{dependent_var}_boxplot_{dependent_var}.{format}".lower().replace(" ", "_"))
        plt.clf()