import seaborn as sns
import re
from matplotlib import pyplot as plt
from pathlib import Path
from loader import dependent_variables, load_data

def generate_boxplot(data: str, output: str, format="png"):
    pattern = r"^[a-zA-Z0-9-]+$"
    
    # Use re.match to check if the extension matches the pattern
    if not re.match(pattern, format):
        raise ValueError(f"invalid given format: {format}")

    output_path: Path = Path(output) if output else Path("figures")
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    # Set figure size to A4 landscape (in inches)
    a4_width_inch = 11.69  # 29.7 cm
    a4_height_inch = 8.27  # 21.0 cm

    plt.figure(figsize=(a4_width_inch, a4_height_inch))

    df = load_data(data)
    for independent_var in ["algorithm"]:
        for dependent_var in dependent_variables:
            ax = sns.boxplot(data=df, y=dependent_var, x=independent_var, hue="map")
            ax.set(yscale="log")
            plt.savefig(output_path / f"{independent_var}_{dependent_var}.{format}")
            plt.clf()
