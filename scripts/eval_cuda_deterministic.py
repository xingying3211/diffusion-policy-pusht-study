"""Run the official LeRobot evaluator with strict loading and CUDA evidence."""
import atexit
import json
import os
import time
from pathlib import Path

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.scripts.eval import main
import lerobot.scripts.eval as eval_module

result_dir = Path(os.environ["RUN_DIR"])
result_dir.mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available(), "CUDA unavailable: refusing CPU fallback"
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
calls = 0
strict_loaded = False
first_trace = None
original_load = DiffusionPolicy.from_pretrained.__func__
original_select = DiffusionPolicy.select_action


@classmethod
def strict_load(cls, *args, **kwargs):
    global strict_loaded
    kwargs["strict"] = True
    policy = original_load(cls, *args, **kwargs)
    assert next(policy.parameters()).device.type == "cuda"
    strict_loaded = True
    print("CUDA_EVIDENCE: checkpoint loaded strictly; model parameters on CUDA", flush=True)
    return policy


def select_with_evidence(self, batch):
    global calls, first_trace
    for key in ("observation.image", "observation.state"):
        assert batch[key].device.type == "cuda", (key, batch[key].device)
    queues = getattr(self, "_queues", None)
    queued_before = len(queues["action"]) if queues and "action" in queues else 0

    action = original_select(self, batch)
    assert action.device.type == "cuda", action.device
    calls += 1
    record = {
    "step": calls,
    "replanned": queued_before == 0,
    "state": batch["observation.state"].detach().cpu().flatten().tolist(),
    "action": action.detach().cpu().flatten().tolist(),
    }

    with (result_dir / "action_step_trace.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    if first_trace is None:
        first_trace = {
            "model_device": str(next(self.parameters()).device),
            "observations": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype),
                      "device": str(value.device), "min": value.min().item(),
                      "max": value.max().item()}
                for key, value in batch.items() if isinstance(value, torch.Tensor)
            },
            "state": batch["observation.state"].cpu().tolist(),
            "action": {"shape": list(action.shape), "dtype": str(action.dtype),
                       "device": str(action.device), "value": action.cpu().tolist()},
            "policy": {"n_obs_steps": self.config.n_obs_steps,
                       "horizon": self.config.horizon,
                       "n_action_steps": self.config.n_action_steps,
                       "num_train_timesteps": self.config.num_train_timesteps,
                       "use_amp": self.config.use_amp},
        }
        (result_dir / "observation_action_trace.json").write_text(json.dumps(first_trace, indent=2))
        print("CUDA_EVIDENCE: real observation and action captured", flush=True)
    return action


def save_evidence():
    evidence = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "strict_checkpoint_loaded": strict_loaded,
        "select_action_calls": calls,
        "cuda_observations_and_actions_verified": calls > 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "process_elapsed_seconds": time.perf_counter() - started,
        "eval_info_exists": (result_dir / "eval_info.json").exists(),
    }
    (result_dir / "gpu_evidence.json").write_text(json.dumps(evidence, indent=2))


# 保存每一步 Reward，不修改模型或评估逻辑
original_rollout = eval_module.rollout
rollout_index = 0

def rollout_with_reward(*args, **kwargs):
    global rollout_index
    result = original_rollout(*args, **kwargs)

    # 对当前 batch_size=1 的实验，提取唯一环境的奖励序列
    rewards = result["reward"].detach().cpu().tolist()

    if len(rewards) != 1:
        raise ValueError("This logger expects eval.batch_size=1")

    # Curation fix: keep each episode, while retaining the legacy last-episode file.
    episode_output = result_dir / f"reward_trace_episode_{rollout_index:03d}.json"
    episode_output.write_text(json.dumps(rewards[0], indent=2), encoding="utf-8")
    rollout_index += 1
    output = result_dir / "reward_trace.json"
    output.write_text(
        json.dumps(rewards[0], indent=2),
        encoding="utf-8"
    )

    print(f"REWARD_TRACE: saved to {output}", flush=True)
    return result

# 保存 LeRobot 原来的随机种子设置函数
original_set_seed = eval_module.set_seed

def deterministic_set_seed(seed):
    # 先执行 LeRobot 原本的种子设置
    original_set_seed(seed)

    # 关闭 cuDNN 自动算法性能测试
    torch.backends.cudnn.benchmark = False

    # 要求 cuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True

    # 要求 PyTorch 使用确定性算法
    torch.use_deterministic_algorithms(True)

    print("DETERMINISM: deterministic mode enabled", flush=True)

# 让 LeRobot 调用我们包装后的函数
eval_module.set_seed = deterministic_set_seed

DiffusionPolicy.from_pretrained = strict_load
DiffusionPolicy.select_action = select_with_evidence
eval_module.rollout = rollout_with_reward
atexit.register(save_evidence)
if __name__ == "__main__":
    main()
