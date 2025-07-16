import io
import math

import matplotlib.pyplot as plt
import pandas as pd


def display_die_stats(die_csv_path: str, fig_name: str):
    labels_of_interest = ["Compiler", "Linker", "Tool", "Format", "Packer", "Archive", "Virus", "Protector"]
    label_lines = {label: [] for label in labels_of_interest}
    df_of_interest = {}
    with open(die_csv_path, "r") as file:
        for line in file:
            line = line.strip()
            for label in labels_of_interest:
                if line.startswith(label):
                    label_lines[label].append(line)
                    break

    n_subplots_cols = math.ceil(math.sqrt(len(labels_of_interest)))

    fig = plt.figure(figsize=(20, 20))

    for i, label in enumerate(label_lines):
        if label != "Packer":
            continue
        if not label_lines:
            continue
        print("====================================")
        print(f"=== For label {label:<10}, got {len(label_lines[label]):>6d} lines")
        print("====================================")

        if len(label_lines[label]) == 0:
            continue

        df_of_interest[label] = pd.read_csv(io.StringIO("\n".join(label_lines[label])), delimiter=';')
        label_distribution = df_of_interest[label].iloc[:, 1].value_counts()
        print(label_distribution)

        plt.subplot(n_subplots_cols, n_subplots_cols, i + 1)
        plt.title(f"Label `{label}` present in {len(label_lines[label])} files")
        plt.barh(label_distribution.keys(), label_distribution, label=label)

    # plt.savefig(f"{fig_name}.pdf", bbox_inches="tight")




if __name__ == "__main__":
    display_die_stats("/opt/work/bd/BODMAS_analysis/info/die/_all.csv")
