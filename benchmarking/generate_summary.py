from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from markdown_table_generator import generate_markdown, table_from_string_list
from utils import create_plot_for_benchmark, create_table_for_benchmark

# Read mean/std and trajectories


# Create table via utils
# Save table
# Plot trajectories via utils
# Save traj

def create_table(results: dict) -> str:
    md_file = ""
    for benchmark in results:
        md_file += f"## {benchmark}\n"
        table = create_table_for_benchmark(results[benchmark])
        table = table_from_string_list(table)
        md_table = generate_markdown(table)
        md_file += md_table + "\n"
    return md_file

def main():
    results_dict = defaultdict(lambda: defaultdict(pd.DataFrame))
    base_result_path = Path("benchmarking/results")
    for file_path in base_result_path.glob("**/traj.parquet.gzip"):
        version = file_path.parts[-3]
        benchmark = file_path.parts[-2]

        result_df = pd.read_parquet(file_path)

        results_dict[benchmark][version] = result_df

    # Convert defaultdict to dict
    results_dict = {k: dict(v) for k, v in results_dict.items()}

    md_table = create_table(results_dict)
    markdown_path = base_result_path / "benchmark_results.md"
    with markdown_path.open("w") as f:
        f.write(md_table)

    for benchmark, data in results_dict.items():
        create_plot_for_benchmark(data, base_result_path, benchmark)


if __name__ == "__main__":
    main()
