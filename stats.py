import argparse
import boxplot
from loader import load_data
import seaborn as sns
import matplotlib.pyplot as plt
from hypothesis import anova

# Function for the "graph:boxplot" subcommand
def graph_boxplot(file, output, extension):
    boxplot.generate_boxplot(file, output, extension)
    # Implement boxplot generation here
    print("Generating boxplots from data...")

# Function for the "hypothesis:anova" subcommand
def hypothesis_anova(args):
    # Implement ANOVA test here
    print("Performing ANOVA test...")

def main():

    # df = load_data("data_0_0.txt")
    # print(anova(df.loc["units traveled"]))

    # return 0
    parser = argparse.ArgumentParser(description="Hypothesis Testing and Graph Generation Script")

    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # Subparser for hypothesis:anova
    parser_anova = subparsers.add_parser("hypothesis:anova", help="Perform ANOVA on data")

    # Subparser for graph:boxplot
    parser_boxplot = subparsers.add_parser("graph:boxplot", help="Generate a boxplot graph from a file")
    parser_boxplot.add_argument("file", help="Input data file for the boxplot")
    parser_boxplot.add_argument("--output", "-o", help="Destination graph folder", default=None)
    parser_boxplot.add_argument("--extension", "-e", help="Graph file format to save", default=None)

    # Subparser for graph:boxplot
    parser_boxplot = subparsers.add_parser("graph:stochastic-boxplot", help="Generate a boxplot graph from a file")
    parser_boxplot.add_argument("file", help="Input data file for the boxplot")
    parser_boxplot.add_argument("--output", "-o", help="Destination graph folder", default=None)
    parser_boxplot.add_argument("--extension", "-e", help="Graph file format to save", default=None)

    parser_boxplot = subparsers.add_parser("graph:perfomance", help="Generate a boxplot graph from a file")
    parser_boxplot.add_argument("file", help="Input data file for the boxplot")
    parser_boxplot.add_argument("--output", "-o", help="Destination graph folder", default=None)
    parser_boxplot.add_argument("--extension", "-e", help="Graph file format to save", default=None)

    parser_boxplot = subparsers.add_parser("graph:rotation-boxplot", help="Generate a boxplot graph from a file")
    parser_boxplot.add_argument("file", help="Input data file for the boxplot")
    parser_boxplot.add_argument("--output", "-o", help="Destination graph folder", default=None)
    parser_boxplot.add_argument("--extension", "-e", help="Graph file format to save", default=None)

    args = parser.parse_args()

    if args.command == "hypothesis:anova":
        hypothesis_anova()
    elif args.command == "graph:boxplot":
        graph_boxplot(args.file, args.output, args.extension)
    elif args.command == "graph:stochastic-boxplot":
        boxplot.generate_hypothesis_1_boxplot(args.file, args.output, args.extension)
    elif args.command == "graph:perfomance":
        boxplot.generate_hypothesis_2_boxplot(args.file, args.output, args.extension)
    elif args.command == "graph:rotation-boxplot":
        boxplot.generate_hypothesis_3_boxplot(args.file, args.output, args.extension)
    else:
        print("Invalid command. Use 'hypothesis:anova' or 'graph:boxplot <file>'.")

if __name__ == "__main__":
    main()
