import os
import re
import pandas as pd
from enums import Algorithm, Map, get_algorithm_name
from pathlib import Path

dependent_variables = [
        "Units traveled",
        "Error",
        "Rotation accumulator",
        "Time taken"
        ]

independent_variables = [
    "Algorithm",
    "Map"
]

unit_mapping = {
    "Units traveled": "grid",
    "Error": "grid re-visits",
    "Rotation accumulator": "°",
    "Time taken": "s"
}

def get_graph_ylabel(variable: str) -> str:
    return f"{variable} [{unit_mapping[variable]}]"

def get_files(root: str = "."):
    # Define the pattern for the file names
    pattern = re.compile(r"data_(\d+)_(\d+)\.txt")

    # Get the list of files in the current directory
    current_directory = Path(root)

    files = [os.path.basename(root)] if not current_directory.is_dir() else os.listdir(root)

    matches = [pattern.match(file) for file in files if pattern.match(file)]
    return [(match.group(0), int(match.group(1)), int(match.group(2))) for match in matches]

def load_data(root: str = "."):
    database = pd.DataFrame(columns=dependent_variables+independent_variables)
    for file, x, y in get_files(root):
        df = pd.read_csv(file, names=dependent_variables+independent_variables)
        df["Algorithm"] = df["Algorithm"].apply(lambda x: get_algorithm_name(Algorithm(x)))
        df["Map"] = df["Map"].apply(lambda x: Map(x).name)
        df["x"] = x
        df["y"] = y
        database = pd.concat((df, database))
    
    database = database.loc[~(database["Algorithm"].isin([
        get_algorithm_name(Algorithm.AStarOrientation),
        # get_algorithm_name(Algorithm.AStarRandom),
        get_algorithm_name(Algorithm.AStarSequential)
    ])), :]

    df = pd.read_csv("data_compare_astar.txt", names=dependent_variables + independent_variables)
    df["Algorithm"] = df["Algorithm"].apply(lambda x: get_algorithm_name(Algorithm(x)))
    df["Map"] = df["Map"].apply(lambda x: Map(x).name)
    df["x"] = 0
    df["y"] = 0

    database = pd.concat((database, df))
    return database
