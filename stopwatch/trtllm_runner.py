from typing import Any, Mapping, Optional
import contextlib
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
import warnings

import modal

from .resources import app, hf_cache_volume, hf_secret


HF_CACHE_PATH = "/cache"
SCALEDOWN_WINDOW = 2 * 60  # 2 minutes
STARTUP_TIMEOUT = 30 * 60  # 30 minutes
TIMEOUT = 60 * 60  # 1 hour


def trtllm_image_factory(
    trtllm_version: str = "0.17.0.post1", cuda_version: str = "12.8.0"
):
    return (
        modal.Image.from_registry(
            f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04",
            add_python="3.12",  # TRT-LLM requires Python 3.12
        )
        .entrypoint([])  # remove verbose logging by base image on entry
        .apt_install("openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget")
        .pip_install(
            f"tensorrt-llm=={trtllm_version}",
            "pynvml<12",  # avoid breaking change to pynvml version API
            pre=True,
            extra_index_url="https://pypi.nvidia.com",
        )
        .pip_install(
            "hf-transfer==0.1.9",
            "huggingface_hub==0.28.1",
            "fastapi",
            "numpy",
            "packaging",
            "SQLAlchemy",
        )
        .env(
            {
                "HF_HUB_CACHE": HF_CACHE_PATH,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
    )


def trtllm_cls(
    image=trtllm_image_factory(),
    secrets=[hf_secret],
    gpu="H100!",
    volumes={HF_CACHE_PATH: hf_cache_volume},
    cpu=4,
    memory=65536,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    region="us-ashburn-1",
):
    def decorator(cls):
        return app.cls(
            image=image,
            secrets=secrets,
            gpu=gpu,
            volumes=volumes,
            cpu=cpu,
            memory=memory,
            allow_concurrent_inputs=1000000,  # Set to a high number to prevent auto-scaling
            scaledown_window=scaledown_window,
            timeout=timeout,
            region=region,
        )(cls)

    return decorator


class trtLLMBase:
    """
    A Modal class that runs an OpenAI-compatible trtLLM server.
    """

    def construct_engine_kwargs(self) -> dict:
        import torch
        import tensorrt_llm
        from tensorrt_llm import BuildConfig
        from tensorrt_llm.llmapi import (
            QuantConfig,
            KvCacheConfig,
            CalibConfig,
            LookaheadDecodingConfig,
        )
        from tensorrt_llm.plugin.plugin import PluginConfig

        print("converting engine kwarg strings into config objects")

        config_name_to_constructor = {
            "quant_config": lambda x: QuantConfig(**x),
            "kv_cache_config": lambda x: KvCacheConfig(**x),
            "calib_config": lambda x: CalibConfig(**x),
            "build_config": lambda x: BuildConfig(**x),
            "plugin_config": PluginConfig.from_dict,
            # TODO: SchedulerConfig
            # TODO: Add support for other decoding configs
            "speculative_config": lambda x: LookaheadDecodingConfig(**x),
        }

        engine_kwargs = json.loads(self.server_config).get("llm_kwargs", {})
        if "tensor_parallel_size" not in engine_kwargs:
            engine_kwargs["tensor_parallel_size"] = torch.cuda.device_count()
        assert engine_kwargs["tensor_parallel_size"] == torch.cuda.device_count()

        self.config_fingerprint = self.get_config_fingerprint(
            tensorrt_llm.__version__, engine_kwargs
        )

        def construct_configs(x):
            for key, value in x.items():
                if isinstance(value, dict):
                    x[key] = construct_configs(value)
                if key in config_name_to_constructor:
                    x[key] = config_name_to_constructor[key](value)
            return x

        engine_kwargs = construct_configs(engine_kwargs)

        self.lookahead_config = engine_kwargs.get("speculative_config")

        print("number of GPUs:", engine_kwargs["tensor_parallel_size"])
        assert engine_kwargs["tensor_parallel_size"] == torch.cuda.device_count()

        return engine_kwargs

    def build_engine(self, engine_path, engine_kwargs) -> None:
        from tensorrt_llm import LLM

        print(f"building new engine at {engine_path}")
        llm = LLM(model=self.model_path, **engine_kwargs)
        llm.save(engine_path)
        del llm

    @modal.enter()
    def enter(self):
        from huggingface_hub import snapshot_download
        from tensorrt_llm import LLM

        print("Received server config:")
        print(self.server_config)

        print("downloading base model if necessary")
        self.model_path = snapshot_download(self.model)

        engine_kwargs = self.construct_engine_kwargs()

        # FIXME just make this trttlm version-engine
        engine_path = os.path.join(
            self.model_path, "trtllm_engine", self.config_fingerprint
        )
        if not os.path.exists(engine_path):
            self.build_engine(engine_path, engine_kwargs)

        print(f"loading engine from {engine_path}")
        self.llm = LLM(model=engine_path, **engine_kwargs)

    @modal.asgi_app()
    def start(self):
        from transformers import AutoTokenizer
        from tensorrt_llm.serve import OpenAIServer
        from fastapi.responses import JSONResponse

        """Start a trtLLM server."""

        assert self.model, "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        server = OpenAIServer(llm=self.llm, model=self.model, hf_tokenizer=tokenizer)

        async def get_iteration_stats(self) -> JSONResponse:
            stats = []
            return JSONResponse(content=stats)
            # async for stat in self.llm.get_stats_async(2):
            # stats.append(stat)
            # return JSONResponse(content=stats)

        server.app.add_api_route("/metrics", get_iteration_stats, methods=["GET"])

        return server.app

    @modal.exit()
    def shutdown(self):
        del self.llm

    def get_config_fingerprint(self, trtllm_version, config_kwargs):
        """Hash config kwargs so we can cache the built engine."""
        return (
            trtllm_version
            + "-"
            + self.model
            + "-"
            + hashlib.md5(
                json.dumps(config_kwargs, sort_keys=True).encode()
            ).hexdigest()
        )


@trtllm_cls(region="us-chicago-1")
class trtLLM(trtLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@contextlib.contextmanager
def trtllm(
    model: str,
    gpu: str,
    region: str,
    server_config: Optional[Mapping[str, Any]] = None,
    profile: bool = False,
):
    if profile:
        raise ValueError("Profiling is not supported for trtLLM")

    warnings.warn(
        "GPU and region selection are not yet supported for trtLLM. Spinning up an H100 in us-chicago-1..."
    )

    extra_query = {
        "model": model,
        # Sort keys to ensure that this parameter doesn't change between runs
        # with the same vLLM configuration
        "server_config": json.dumps(server_config, sort_keys=True),
        "caller_id": modal.current_function_call_id(),
    }

    cls = trtLLM

    args = urllib.parse.urlencode(extra_query)
    url = cls(model="").start.web_url

    # Wait for trtLLM server to start
    response_code = -1
    print(f"Requesting health at {url}/health?{args}")

    while response_code != 200:
        response = urllib.request.urlopen(f"{url}/health?{args}")
        response_code = response.code
        time.sleep(5)

    print("Connected to trtLLM instance")
    yield (url, extra_query)
