"""Reward functions for RL fine-tuning of translation / generation models
(GRPO, PPO, RLOO, best-of-n). Scores are exact sacreBLEU sentence scores,
computed natively and in parallel for the whole batch.

TRL example::

    from trl import GRPOTrainer
    from swadesh.rewards import chrf_reward

    trainer = GRPOTrainer(..., reward_funcs=[chrf_reward(ref_column="reference")])

Plain example::

    reward = chrf_reward()
    reward(completions=["Das ist gut."], reference=["Das ist sehr gut."])  # -> [..]
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from .metrics import BLEU, CHRF


def _texts(completions) -> List[str]:
    out = []
    for c in completions:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list) and c and isinstance(c[-1], dict):  # conversational format
            out.append(c[-1].get("content", ""))
        elif isinstance(c, dict):
            out.append(c.get("content", ""))
        else:
            out.append(str(c))
    return out


def _make(metric, ref_column: str, scale: float, name: str) -> Callable:
    def reward(completions: Sequence, **kwargs) -> List[float]:
        refs = kwargs.get(ref_column)
        if refs is None:
            raise KeyError(f"reward needs the dataset column {ref_column!r} (got columns {sorted(kwargs)})")
        hyps = _texts(completions)
        streams = [list(refs)] if refs and isinstance(refs[0], str) else [list(s) for s in zip(*refs)]
        return [s * scale for s in metric.sentence_scores(hyps, streams)]

    reward.__name__ = name
    return reward


def chrf_reward(ref_column: str = "reference", scale: float = 0.01, word_order: int = 0, **chrf_kwargs) -> Callable:
    """chrF (or chrF++ with ``word_order=2``) reward in [0, 1] by default (``scale=0.01``).
    ``ref_column`` may hold one reference string per example or a list of references."""
    return _make(CHRF(word_order=word_order, **chrf_kwargs), ref_column, scale,
                 "chrf_reward" if word_order == 0 else "chrfpp_reward")


def bleu_reward(ref_column: str = "reference", scale: float = 0.01, tokenize: Optional[str] = None,
                **bleu_kwargs) -> Callable:
    """Sentence-BLEU reward (effective order on by default) in [0, 1] by default."""
    bleu_kwargs.setdefault("effective_order", True)
    return _make(BLEU(tokenize=tokenize, **bleu_kwargs), ref_column, scale, "bleu_reward")
