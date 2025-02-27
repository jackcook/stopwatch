import os
import subprocess
import yaml

import click
import modal

from stopwatch.resources import figures_volume, traces_volume


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--model",
    type=str,
    help="Name of the model to benchmark.",
    required=True,
)
@click.option(
    "--data",
    type=str,
    default="prompt_tokens=512,generated_tokens=128",
    help="The data source to use for benchmarking. Depending on the data-type, it should be a path to a data file containing prompts to run (ex: data.txt), a HuggingFace dataset name (ex: 'neuralmagic/LLM_compression_calibration'), or a configuration for emulated data (ex: 'prompt_tokens=128,generated_tokens=128').",
)
@click.option(
    "--data-type",
    type=str,
    default="emulated",
    help="The type of data to use, such as 'emulated', 'file', or 'transformers'. Defaults to 'emulated'.",
)
@click.option(
    "--gpu",
    type=str,
    default="H100",
    help="GPU to run the vLLM server on. Defaults to 'H100'.",
)
@click.option(
    "--vllm-docker-tag",
    type=str,
    default="latest",
    help="Docker tag to use for vLLM. Defaults to 'latest'.",
)
@click.option(
    "--vllm-env-vars",
    "-e",
    type=str,
    multiple=True,
    default=[],
    help="Environment variables to set on the vLLM server. Each argument should be in the format of 'KEY=VALUE'",
)
@click.option(
    "--vllm-extra-args",
    type=str,
    default="",
    help="Extra arguments to pass to the vLLM server.",
)
def run_benchmark(**kwargs):
    kwargs["vllm_env_vars"] = {
        k: v for k, v in (e.split("=") for e in kwargs["vllm_env_vars"])
    }
    kwargs["vllm_extra_args"] = (
        kwargs["vllm_extra_args"].split(" ") if kwargs["vllm_extra_args"] else []
    )

    f = modal.Function.from_name("stopwatch", "run_benchmark")
    fc = f.spawn(**kwargs)
    print(f"Benchmark running at {fc.object_id}")


@cli.command()
@click.option(
    "--model",
    type=str,
    help="Name of the model to run while profiling vLLM.",
    required=True,
)
@click.option(
    "--num-requests",
    type=int,
    help="Number of requests to make to vLLM while profiling.",
    default=10,
)
def run_profiler(**kwargs):
    f = modal.Function.from_name("stopwatch", "run_profiler")
    fc = f.spawn(**kwargs)
    print(f"Profiler running at {fc.object_id}...")
    trace_path = fc.get()

    # Download and save trace
    os.makedirs("traces", exist_ok=True)
    local_trace_path = os.path.join("traces", trace_path)

    with open(local_trace_path, "wb") as f:
        for chunk in traces_volume.read_file(trace_path):
            f.write(chunk)

    # Optionally show file in Finder
    answer = input(f"Saved to {trace_path}. Show in Finder? [Y/n] ")

    if answer != "n":
        subprocess.run(["open", "-R", local_trace_path])


@cli.command()
@click.argument("config-path", type=str)
def generate_figure(config_path: str):
    f = modal.Function.from_name("stopwatch", "generate_figure")
    fc = f.spawn(**yaml.load(open(config_path), Loader=yaml.SafeLoader))

    print("Generating figure...")
    remote_figure_path = fc.get()

    # Download and save figure
    local_figure_path = config_path.replace(".yaml", ".png")

    with open(local_figure_path, "wb") as f:
        for chunk in figures_volume.read_file(remote_figure_path):
            f.write(chunk)

    # Optionally open figure
    answer = input(f"Saved to {local_figure_path}. Open it? [Y/n] ")

    if answer != "n":
        subprocess.run(["open", local_figure_path])


if __name__ == "__main__":
    cli()
