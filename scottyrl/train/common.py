"""Group rollouts and batch assembly. *** STUDENT IMPLEMENTATION FILE ***

These two functions are shared by the sync (Parts 1-2) and async (Part 3)
trainers.
"""

from __future__ import annotations

import asyncio  # noqa: F401  (student dependency)
from typing import Callable

from scottyrl.advantages import compute_group_advantages  # noqa: F401  (student dependency)
from scottyrl.agents.base import Agent
from scottyrl.agents.xml_tool_call import XMLToolCallAgent
from scottyrl.datagen import trajectory_to_training_sequences  # noqa: F401  (student dependency)
from scottyrl.envs.base import Env
from scottyrl.policy import Policy
from scottyrl.rollout import run_episode  # noqa: F401  (student dependency)
from scottyrl.types import Task, TokenSequence, Trajectory


async def collect_group(
    policy: Policy,
    task: Task,
    env_factory: Callable[[Task], Env],
    group_size: int,
    agent_cls: type[Agent] = XMLToolCallAgent,
) -> list[Trajectory]:
    """Run `group_size` independent rollouts of `task` concurrently.

    Contract: every rollout gets its own fresh env instance and fresh agent
    instance (no shared state between rollouts), and the rollouts run
    concurrently so their LLM calls overlap in time. Each rollout is
    `run_episode(agent_cls(policy), env_factory(task))` (see
    scottyrl.rollout); launch all `group_size` of them with `asyncio.gather`.

    Returns: `group_size` trajectories, in rollout order.
    """
    # TODO(student): implement this function.
    rollouts = [
        run_episode(agent_cls(policy), env_factory(task))
        for _ in range(group_size)
    ]
    return list(await asyncio.gather(*rollouts))


def groups_to_batch(
    groups: list[list[Trajectory]],
    method: str = "grpo",
) -> tuple[list[TokenSequence], dict]:
    """Turn a list of trajectory groups into a flat training batch.

    Contract:
      - Each group's advantages come from your `compute_group_advantages`,
        applied to that group's episode rewards (in rollout order).
      - A degenerate group (advantages None) contributes no sequences and is
        counted, but its rewards still count toward mean_reward.
      - Every remaining trajectory contributes its training sequences (your
        `trajectory_to_training_sequences`) with its own advantage, even
        when that advantage is 0.0.

    Returns: (sequences, stats), where stats has at least these keys:
        "num_groups"              total groups passed in
        "num_degenerate_groups"   groups skipped as degenerate
        "frac_degenerate_groups"  ratio of the two above (0.0 if no groups)
        "mean_reward"             mean episode reward over ALL trajectories,
                                  degenerate groups included (0.0 if there
                                  are none)
        "num_sequences"           len(sequences)
        "num_action_tokens"       total mask==1 positions in the batch
    """
    sequences: list[TokenSequence] = []
    all_rewards: list[float] = []
    num_degenerate_groups = 0

    for group in groups:
        # 一组中的轨迹来自同一道题；优势值只在这一组内比较。
        rewards = [traj.reward for traj in group]

        # 退化组虽然不参与训练，其奖励仍计入总体平均奖励。
        all_rewards.extend(rewards)

        advantages = compute_group_advantages(rewards, method=method)
        if advantages is None:
            num_degenerate_groups += 1
            continue

        # advantages 与 group 顺序一致：每条轨迹使用自己的优势值。
        # 一条轨迹可能因序列拓展中断而产生多条训练序列。
        for traj, advantage in zip(group, advantages):
            sequences.extend(
                trajectory_to_training_sequences(traj, advantage)
            )
    num_groups = len(groups)
    stats = {
        "num_groups": num_groups,
        "num_degenerate_groups": num_degenerate_groups,
        "frac_degenerate_groups": (
            num_degenerate_groups / num_groups if num_groups else 0.0
        ),
        "mean_reward": (
            sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        ),
        "num_sequences": len(sequences),
        # mask==1 的位置才是参与训练的模型生成 token。
        "num_action_tokens": sum(
            sequence.num_action_tokens for sequence in sequences
        ),
    }
    return sequences, stats
