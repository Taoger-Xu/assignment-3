# Local SkyRL server for Part 2

The assignment client still uses the Tinker API. This launcher runs the
assignment's synchronous Number Search backend on two local GPUs, using the
same backend settings and pinned SkyRL commit as the Modal launcher.
On this host, the launcher disables NCCL peer-to-peer transport because a
two-GPU all-reduce stalls with it enabled; shared-memory transport passed a
two-GPU communication check.

The commands below assume the SkyRL checkout, virtual environment and state
directory are siblings of this assignment repository. Run the client commands
from the assignment root.

```bash
git clone https://github.com/adityasoni9998/SkyRL.git ../skyrl-local
git -C ../skyrl-local checkout 491e1329c314ef43818fb6824d9f2ae61883416c
git -C ../skyrl-local apply ../assignment-3/skyrl_local/skyrl_gpu_uuid.patch
cd ../skyrl-local
UV_PROJECT_ENVIRONMENT=../skyrl-venv UV_LINK_MODE=copy uv sync --extra tinker --extra megatron
cd ../assignment-3

python3 skyrl_local/serve_number_search_sync.py \
  --skyrl-dir ../skyrl-local --venv ../skyrl-venv \
  --state-dir ../skyrl-state/ns-sync --gpus 1,2
```

Keep the server terminal open. In another terminal:

```bash
export TINKER_BASE_URL=http://127.0.0.1:9000
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml \
  --set train.num_steps=10 --set train.eval_every=2 \
  --set train.eval_tasks=20 --set train.run_name=probe_p2b_local

# Once the probe succeeds, run the required configs unchanged, one at a time:
uv run python scripts/train_sync.py --config configs/p2a_number_search_reinforce.yaml
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml
```

Use `nvidia-smi` to confirm the selected GPUs are free before starting. The
model weights are cached in `HF_HOME` (default `~/.cache/huggingface`), and
checkpoints and the Tinker database stay under `--state-dir`. Restart with
the same state directory to resume a run. Stop the server with Ctrl-C after
finishing the experiments.
