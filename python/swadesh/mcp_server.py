"""MCP server: sacreBLEU-exact BLEU / chrF / TER as tools for AI agents.

    pip install "swadesh[mcp]"
    swadesh-mcp                      # stdio transport (Claude Code, Claude Desktop, Cursor...)

Claude Code:  claude mcp add swadesh -- swadesh-mcp

Every number returned is identical to what sacreBLEU 2.6.0 computes, and each
corpus score carries sacreBLEU's reproducibility signature.
"""

from __future__ import annotations

from typing import List, Optional

import swadesh

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError as _MCPToolError
except ImportError as e:  # pragma: no cover
    raise ImportError('The MCP server needs the "mcp" package: pip install "swadesh[mcp]"') from e

server = MCPServer(
    name="swadesh",
    title="swadesh: sacreBLEU-exact MT metrics",
    description="BLEU, chrF, chrF++ and TER, bit-identical to sacreBLEU 2.6.0, computed in Rust.",
    instructions=(
        "Use these tools to evaluate machine translation or other generated text against references. "
        "Scores are identical to sacreBLEU, so they can be reported with the returned signature. "
        "`references` is a list of reference streams: one list per reference set, each the same length "
        "as `hypotheses`."
    ),
    version=swadesh.__version__,
)

_METRICS = {"bleu", "chrf", "chrf++", "ter"}


class ToolError(_MCPToolError, ValueError):
    """Bad tool arguments. MCP forwards the message to the agent (an anticipated
    tool failure), and Python callers can still catch it as a ValueError."""


def _metric(name: str, tokenize: Optional[str], lowercase: bool):
    name = name.lower()
    if name == "bleu":
        return swadesh.BLEU(tokenize=tokenize, lowercase=lowercase)
    if name == "chrf":
        return swadesh.CHRF(lowercase=lowercase)
    if name == "chrf++":
        return swadesh.CHRF(word_order=2, lowercase=lowercase)
    if name == "ter":
        return swadesh.TER()  # sacreBLEU's default: case-insensitive
    raise ToolError(f"unknown metric {name!r}; choose from {sorted(_METRICS)}")


def _check(hypotheses: List[str], references: List[List[str]]):
    if not hypotheses:
        raise ToolError("hypotheses is empty")
    if not references or any(len(r) != len(hypotheses) for r in references):
        raise ToolError("references must be a list of streams, each as long as hypotheses")


@server.tool()
def corpus_score(
    hypotheses: List[str],
    references: List[List[str]],
    metrics: List[str] = ["bleu", "chrf"],  # noqa: B006
    tokenize: Optional[str] = None,
    lowercase: bool = False,
    confidence: bool = False,
) -> dict:
    """Corpus-level BLEU / chrF / chrF++ / TER, identical to sacreBLEU, with signatures.

    tokenize (BLEU only): 13a (default), intl, zh, char, none, ja-mecab, ko-mecab, spm, flores200.
    confidence: add a 95% bootstrap confidence interval (1000 resamples, sacreBLEU's procedure).
    """
    _check(hypotheses, references)
    out = {}
    for name in metrics:
        m = _metric(name, tokenize, lowercase)
        score = m.corpus_score(hypotheses, references, n_bootstrap=1000 if confidence else 1)
        entry = {"score": score.score, "formatted": str(score), "signature": str(m.get_signature())}
        if confidence:
            entry["mean"], entry["ci95"] = float(score._mean), float(score._ci)
        out[name] = entry
    return out


@server.tool()
def sentence_scores(
    hypotheses: List[str],
    references: List[List[str]],
    metric: str = "chrf",
    tokenize: Optional[str] = None,
    lowercase: bool = False,
) -> dict:
    """Per-segment scores (BLEU uses effective order, as recommended for sentence-level BLEU).
    Useful for finding the worst translations or for filtering data."""
    _check(hypotheses, references)
    name = metric.lower()
    if name == "bleu":
        return {"scores": swadesh.BLEU(tokenize=tokenize, lowercase=lowercase, effective_order=True)
                .sentence_scores(hypotheses, references)}
    if name in ("chrf", "chrf++"):
        return {"scores": _metric(name, None, lowercase).sentence_scores(hypotheses, references)}
    if name == "ter":
        m = swadesh.TER()
        return {"scores": [m.sentence_score(h, [r[i] for r in references]).score for i, h in enumerate(hypotheses)]}
    raise ToolError(f"unknown metric {metric!r}")


@server.tool()
def mbr_select(candidates: List[str], metric: str = "chrf") -> dict:
    """Minimum Bayes risk selection: pick the candidate that agrees most with the others
    (each scored against all other candidates as pseudo-references)."""
    if len(candidates) < 2:
        raise ToolError("need at least two candidates")
    m = swadesh.BLEU(effective_order=True) if metric.lower() == "bleu" else _metric(metric, None, False)
    best, util = swadesh.mbr(candidates, metric=m)
    return {"best_index": best, "best": candidates[best], "expected_utility": [float(u) for u in util]}


@server.tool()
def tokenize(lines: List[str], tokenizer: str = "13a") -> dict:
    """Tokenize exactly like sacreBLEU's tokenizer (13a, intl, zh, char, none)."""
    return {"lines": swadesh.tokenize(lines, tokenizer)}


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
