# stopwatch

_A simple solution for benchmarking [vLLM](https://docs.vllm.ai/en/latest/) and [trtLLM](https://github.com/NVIDIA/TensorRT-LLM) on [Modal](https://modal.com/) with [guidellm](https://github.com/neuralmagic/guidellm)._ ⏱️

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

## Run benchmark

To run a single benchmark, you can use the `run-benchmark` command.
For example, to run a constant-rate benchmark at 5 queries per second:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
OUTPUT_PATH=results.json

modal run -w $OUTPUT_PATH cli.py::run_benchmark --model $MODEL --rate-type constant --rate 5
```

You can then view the results of the benchmark in your newly created JSON file.

## Run and plot multiple benchmarks

To run multiple benchmarks at once, first deploy the project:

```bash
modal deploy -m stopwatch
```

Then, call the function remotely:

To run multiple benchmarks at once, you can use the `run-benchmark-function` command, along with a configuration file.

```bash
python cli.py run-benchmark-suite configs/data-distributions.yaml
```

Once the suite has finished, you will be prompted to open a link to a [Datasette](https://datasette.io/){:target="\_blank"} UI with your results.

## Run profiler

To profile vLLM with the PyTorch profiler, use the following command:

```bash
python cli.py run-profiler --model meta-llama/Llama-3.1-8B-Instruct --num-requests 10
```

Once the profiling is done, you will be prompted to download the generated trace and reveal it in Finder.
Keep in mind that generated traces can get very large, so it is recommended to only send a few requests while profiling.
Traces can then be visualized at [https://ui.perfetto.dev](https://ui.perfetto.dev).

## License

Stopwatch is available under the MIT license. See the [LICENSE](/LICENSE.md) file for more details.
