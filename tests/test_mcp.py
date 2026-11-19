"""The MCP tools must return exactly sacreBLEU's numbers."""
import asyncio
import json

import pytest
import sacrebleu

mcp = pytest.importorskip("mcp")
from swadesh.mcp_server import server  # noqa: E402

H = ["The cat sat on the mat.", "A dog barked loudly.", "Prices rose by 3.2% in October."]
R = [["The cat is on the mat.", "The dog barked.", "Prices increased 3.2% in October."]]


def call(name, args):
    res = asyncio.run(server.call_tool(name, args))
    assert not res.is_error, res
    return json.loads(res.content[0].text)


def test_corpus_score_exact():
    out = call("corpus_score", {"hypotheses": H, "references": R, "metrics": ["bleu", "chrf", "chrf++", "ter"]})
    assert out["bleu"]["score"] == sacrebleu.corpus_bleu(H, R).score
    assert out["chrf"]["score"] == sacrebleu.corpus_chrf(H, R).score
    assert out["chrf++"]["score"] == sacrebleu.CHRF(word_order=2).corpus_score(H, R).score
    assert out["ter"]["score"] == sacrebleu.corpus_ter(H, R).score


def test_sentence_scores_exact():
    got = call("sentence_scores", {"hypotheses": H, "references": R, "metric": "chrf"})
    m = sacrebleu.CHRF()
    assert got["scores"] == [m.sentence_score(h, [r]).score for h, r in zip(H, R[0])]


def test_mbr_and_tokenize():
    out = call("mbr_select", {"candidates": ["the cat sat", "a cat sat", "the cat sat down", "dogs"]})
    assert out["best"] in ("the cat sat", "the cat sat down")
    tok = call("tokenize", {"lines": ["Hello, world!"], "tokenizer": "13a"})
    assert tok["lines"] == ["Hello , world !"]
