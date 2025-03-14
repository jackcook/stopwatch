import modal

from .benchmark import BenchmarkDefaults, get_benchmark_fingerprint
from .resources import app, datasette_volume, results_dict, results_volume
from .run_benchmark import all_benchmark_runner_classes


DATASETTE_PATH = "/datasette"
RESULTS_PATH = "/results"
TIMEOUT = 60 * 60  # 1 hour


benchmark_suite_image = modal.Image.debian_slim().pip_install("pandas", "sqlite-utils")

with benchmark_suite_image.imports():
    from typing import Any, Dict, List
    import copy
    import itertools
    import json
    import os
    import shutil
    import tempfile

    import numpy as np
    import pandas as pd
    import sqlite_utils


def histogram_median(bins, counts):
    if len(bins) == len(counts) == 0:
        return None

    assert len(bins) == len(counts) + 1, f"len({bins}) != len({counts}) + 1"

    total = sum(counts)
    half = total / 2

    # Walk through the histogram until we reach or exceed the halfway point
    cumulative = 0
    for i, count in enumerate(counts):
        new_cumulative = cumulative + count

        if new_cumulative >= total / 2:
            # Linearly interpolate within the bin
            fraction = (half - cumulative) / count if count > 0 else 0
            bin_width = bins[i + 1] - bins[i]
            return bins[i] + fraction * bin_width

        cumulative = new_cumulative


@app.function(
    image=benchmark_suite_image,
    volumes={DATASETTE_PATH: datasette_volume, RESULTS_PATH: results_volume},
    timeout=TIMEOUT,
)
def run_benchmark_suite(
    benchmarks: List[Dict[str, Any]],
    suite_id: str = "stopwatch",
    repeats: int = 1,
    recompute: bool = False,
):
    assert repeats > 0

    benchmarks_to_run = []

    # Create fingerprints for each benchmark. This allows us to check if the
    # benchmark has already been run recently, in which case we don't need
    # to run it again.
    for benchmark in benchmarks:
        for repeat_index in range(repeats):
            repeat_benchmark = copy.deepcopy(benchmark)
            repeat_benchmark["config"]["repeat_index"] = repeat_index
            repeat_benchmark["fingerprint"] = get_benchmark_fingerprint(
                **repeat_benchmark["config"]
            )
            benchmarks_to_run.append(repeat_benchmark)

    # Run benchmarks that aren't already cached
    pending_benchmarks = []

    for benchmark in benchmarks_to_run:
        if benchmark["fingerprint"] in results_dict and not recompute:
            continue

        benchmark_cls = all_benchmark_runner_classes[benchmark["config"]["region"]]
        fc = benchmark_cls().run_benchmark.spawn(**benchmark["config"])
        pending_benchmarks.append(fc)

    # Wait for all newly run benchmarks to finish
    for fc in pending_benchmarks:
        fc.get()

    # Insert into SQLite
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db_path = os.path.join(tmpdir, f"{suite_id}.db")

        db = sqlite_utils.Database(tmp_db_path)
        table = db["benchmarks"]

        for benchmark in benchmarks_to_run:
            results_path = os.path.join(
                RESULTS_PATH, f"{results_dict[benchmark['fingerprint']]}.json"
            )
            results_data = json.load(open(results_path))

            df = pd.DataFrame(results_data["benchmarks"][0]["benchmarks"])
            df = df[df["mode"] != "throughput"]

            # TODO: Handle this automatically
            for _, row in df.iterrows():
                if len(row["results"]) == 0:
                    raise ValueError(
                        f"No results found for benchmark {benchmark['fingerprint']}. Please re-run this benchmark."
                    )

            df["duration"] = df.apply(
                lambda x: x["results"][-1]["end_time"] - x["results"][0]["start_time"],
                axis=1,
            )
            df["requests_per_second"] = df.apply(
                lambda x: len(x["results"]) / x["duration"], axis=1
            )

            # Create distributions
            df["ttft_distribution"] = df.apply(
                lambda x: [
                    y["first_token_time"]
                    for y in x["results"]
                    if y["first_token_time"] is not None
                ],
                axis=1,
            )
            df["ttlt_distribution"] = df.apply(
                lambda x: [
                    x["results"][i]["end_time"] - x["results"][i]["start_time"]
                    for i in range(len(x["results"]))
                ],
                axis=1,
            )
            df["itl_distribution"] = df.apply(
                lambda x: [
                    decode
                    for result in x["results"]
                    for decode in result["decode_times"]["data"]
                ],
                axis=1,
            )

            # Save KV cache usage from vLLM metrics
            df["kv_cache_usage_mean"] = df.apply(
                lambda x: 100
                * np.mean([y["kv_cache_usage"] for y in x["vllm_metrics"]]),
                axis=1,
            )

            # Save TPOT median from vLLM metrics
            df["tpot_median"] = df.apply(
                lambda x: histogram_median(
                    x["vllm_metrics"][-1]["time_per_output_token"]["bins"],
                    x["vllm_metrics"][-1]["time_per_output_token"]["data"],
                ),
                axis=1,
            )

            # Calculate percentiles of time-to-first-token, time-to-last-token,
            # and inter-token latency
            for key, statistic in itertools.product(
                ["itl", "ttft", "ttlt"], ["mean", 50, 90, 95, 99]
            ):
                if statistic == "mean":
                    df[f"{key}_mean"] = df.apply(
                        lambda x: np.mean(x[f"{key}_distribution"]), axis=1
                    )
                else:
                    df[f"{key}_p{statistic}"] = df.apply(
                        lambda x: np.percentile(x[f"{key}_distribution"], statistic),
                        axis=1,
                    )

            # Get number of prompt tokens and generated tokens
            data_config = (
                {
                    k: int(v)
                    for param in benchmark["config"]["data"].split(",")
                    for k, v in [param.split("=")]
                }
                if "data" in benchmark["config"]
                else {}
            )

            if "region" not in benchmark["config"]:
                # TODO: Remove this once older benchmark runs are discarded
                benchmark["config"]["region"] = (
                    "us-east4"
                    if benchmark["config"]["gpu"] == "A100-80GB"
                    else "us-ashburn-1"
                )

            table.insert_all(
                [
                    {
                        **benchmark["config"],
                        "id": benchmark["fingerprint"],
                        "gpu": benchmark["config"]
                        .get("gpu", BenchmarkDefaults.GPU)
                        .replace("!", ""),
                        **data_config,
                        **{
                            key: x[key]
                            for key in df.columns
                            if df[key].dtype.kind == "f"
                            and key not in ["duration", "rate"]
                        },
                    }
                    for _, x in df.iterrows()
                ]
            )

        db.close()
        shutil.copyfile(tmp_db_path, os.path.join(DATASETTE_PATH, f"{suite_id}.db"))

    return [benchmark["fingerprint"] for benchmark in benchmarks_to_run]
