"""Run the assignment's synchronous Number Search SkyRL profile on local GPUs.

Install the pinned SkyRL checkout separately, then run this script with that
checkout's environment. The service uses the same backend settings as
skyrl_tinker_modal/serve_number_search_sync.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


BASE_MODEL = "Qwen/Qwen3.5-4B"
SKYRL_COMMIT = "491e1329c314ef43818fb6824d9f2ae61883416c"

BACKEND_CONFIG = {
    "trainer.placement.policy_num_nodes": 1,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
    "trainer.policy.megatron_config.context_parallel_size": 1,
    "trainer.policy.language_model_only": True,
    "generator.inference_engine.language_model_only": True,
    "trainer.policy.megatron_config.lora_config.merge_lora": False,
    "trainer.policy.optimizer_config.max_grad_norm": 1.0,
    "trainer.policy.optimizer_config.adam_betas": [0.9, 0.95],
    "trainer.policy.optimizer_config.weight_decay": 0.0,
    "trainer.use_expandable_segments": True,
    "trainer.max_tokens_per_microbatch": 32768,
    "generator.inference_engine.max_num_batched_tokens": 50000,
    "generator.inference_engine.max_num_seqs": 256,
    "generator.inference_engine.enable_prefix_caching": True,
    "generator.inference_engine.enforce_eager": False,
    "generator.inference_engine.engine_init_kwargs.gdn_prefill_backend": "triton",
    "generator.inference_engine.engine_init_kwargs.compilation_config": {
        "cudagraph_mode": "FULL_DECODE_ONLY"
    },
    "generator.inference_engine.use_expandable_segments": False,
    "generator.inference_engine.enable_ray_prometheus_stats": True,
    "generator.inference_engine.engine_init_kwargs.enable_mfu_metrics": True,
    "trainer.placement.colocate_all": True,
    "trainer.placement.policy_num_gpus_per_node": 2,
    "generator.inference_engine.num_engines": 2,
    "trainer.logprobs_chunk_size": 1024,
    "trainer.fused_lm_head_logprob": True,
    "generator.inference_engine.gpu_memory_utilization": 0.7,
    "generator.inference_engine.engine_init_kwargs.max_model_len": 32768,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skyrl-dir", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="1,2", help="Two local GPU IDs")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    skyrl_dir = args.skyrl_dir.resolve()
    venv = args.venv.resolve()
    state_dir = args.state_dir.resolve()
    if subprocess.check_output(
        ["git", "-C", str(skyrl_dir), "rev-parse", "HEAD"], text=True
    ).strip() != SKYRL_COMMIT:
        parser.error(f"SkyRL checkout must be pinned to {SKYRL_COMMIT}")
    if len(args.gpus.split(",")) != 2:
        parser.error("--gpus must name exactly two devices")
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    gpu_uuids = {
        index.strip(): uuid.strip()
        for row in gpu_rows.splitlines()
        for index, uuid in [row.split(",", 1)]
    }
    try:
        selected_uuids = [gpu_uuids[index.strip()] for index in args.gpus.split(",")]
    except KeyError as exc:
        parser.error(f"Unknown physical GPU index: {exc}")
    if len(set(selected_uuids)) != 2:
        parser.error("--gpus must name two different devices")
    for subdir in ("checkpoints", "database"):
        (state_dir / subdir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(selected_uuids),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        # This host's GPU peer-to-peer path stalls NCCL collectives; SHM works.
        NCCL_P2P_DISABLE="1",
        CUDA_HOME="/usr/local/cuda",
        HF_HOME=env.get("HF_HOME", str(Path.home() / ".cache/huggingface")),
        RAY_RUNTIME_ENV_HOOK="ray._private.runtime_env.uv_runtime_env_hook.hook",
        SKYRL_REPO_ROOT=str(skyrl_dir),
        UV_PROJECT_ENVIRONMENT=str(venv),
        UV_LINK_MODE="copy",
        VIRTUAL_ENV=str(venv),
        PATH=f"{venv / 'bin'}:{env['PATH']}",
    )
    command = [
        "uv", "run", "--active", "--no-sync", "--extra", "tinker", "--extra", "megatron",
        "-m", "skyrl.tinker.api",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--base-model", BASE_MODEL, "--backend", "megatron",
        "--checkpoints-base", str(state_dir / "checkpoints"),
        "--database-url", f"sqlite:///{state_dir / 'database' / 'tinker.db'}",
        "--backend-config", json.dumps(BACKEND_CONFIG),
    ]
    print(
        f"Serving {BASE_MODEL} on physical GPUs {args.gpus} "
        f"({','.join(selected_uuids)}) at http://127.0.0.1:{args.port}",
        flush=True,
    )
    os.execvpe("uv", command, env)


if __name__ == "__main__":
    main()
