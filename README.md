# recall-bench

**Cross-session memory recall benchmark for AI agents.**

A method, not a product. One question: *when your agent's memory stores something today, can it actually find it tomorrow?*

recall-bench measures the cross-session recall capability of any memory system behind an AI agent — SQLite vaults, vector stores, hosted memory APIs — through a reproducible write-then-retrieve protocol with objective hit detection. Zero dependencies. Runs anywhere Python runs.

```
total 20 | hits 2 | recall_rate = 10.0%
```

> **Origin**: built from the LAAP (Living Agent Application Protocol) cognitive architecture practice. The first baseline run against LAAP's production memory vault (2026-08-02) scored **25.0%** with substring (LIKE-style) retrieval. This package generalizes that methodology so any memory system can be measured against the same ruler.

## Why this exists

Two converging judgments from the field:

- **Ilya Sutskever** (2025): humans are not "pre-trained AGI" — *we depend on continual learning*. Real agents learn after deployment, not just during training.
- **Lilian Weng** (2026): context & memory lifecycle will *become part of intelligence itself*, not stay in the software layer.

If memory is becoming a core component of agent intelligence, then **memory deserves a benchmark with the same rigor as reasoning benchmarks**. Most agent teams today cannot answer three basic questions:

1. How much of what my agent stored can it actually retrieve later?
2. Does retrieval degrade as memory grows?
3. Did this week's "memory improvement" actually improve anything?

recall-bench is a first, honest answer to question 1 — and the measurement layer that makes 2 and 3 possible.

## Metrics

| Metric | Definition |
|---|---|
| `recall_rate` | hits / total — the fraction of stored facts retrievable by natural-language query |
| `avg_latency_ms` | mean retrieval latency per query |
| `write_ms_total` | total time to store the probe set |
| `by_domain` | recall rate per topical domain (10 domains) |
| `by_query_len` | recall rate grouped by query length (short ≤10 / long >10) |

## Methodology

The probe protocol keeps measurement **objective and contamination-free**:

1. **20 neutral facts** across 10 topical domains (tech, habits, projects, goals, resources, preferences, events, interests, opinions, efficiency).
2. Each fact carries a **unique marker** (`RB_20260803_07`) appended to its content.
3. The retrieval query is a natural-language paraphrase — **the marker never appears in the query**. A hit is only counted when the marker appears in the returned results. No LLM judge, no subjective scoring.
4. **Two modes**:
   - `clean` — run against an isolated/scratch environment (default)
   - `live` — run against the production store; written records are **auto-cleaned** afterwards via `backend.cleanup()`

Why this design? Because the marker keeps hit detection deterministic, and the paraphrased query keeps the test honest: we measure *semantic* recall, not string copying.

## Quick start

```bash
pip install recall-bench          # once published; or run from source below
python -m recall_bench.cli run    # zero-dependency in-memory backend
```

From source:

```bash
git clone <repo> && cd recall-bench
python -m pytest tests -q         # 6 tests, zero deps
python -m recall_bench.cli run    # clean run against in-memory backend
python -m recall_bench.cli run --output report.json
```

### Measuring YOUR memory system

Implement the `MemoryBackend` protocol (three methods) and point the benchmark at it:

```python
from recall_bench.backends.base import MemoryItem
from recall_bench.runner import run_benchmark

class MyBackend:
    name = "my_mem"
    def start_session(self, title=""): return "s1"
    def store(self, role, content, tags=""): ...          # -> entry id
    def search(self, query, limit=10) -> list[MemoryItem]: ...  # items need .content
    def cleanup(self, ids, session_id=""): ...            # live-mode hygiene

report = run_benchmark(MyBackend(), tag="RB_MINE")
print(report.recall_rate)
```

See `recall_bench/backends/` for a reference in-memory implementation and a real SQLite adapter (`laap_vault`, optional import).

### Custom probe sets

Provide your own facts as JSON — same protocol, your domain:

```json
[
  {"content": "用户喜欢早起", "domain": "习惯", "query": "起床时间"},
  {"content": "项目采用微服务架构", "domain": "项目", "query": "架构风格"}
]
```

```bash
python -m recall_bench.cli run --probe-file my_probe.json
```

## Reference results (2026-08-02/03)

Honest numbers, not marketing:

| Backend | Mode | recall_rate | Notes |
|---|---|---|---|
| `memory` (in-memory substring) | clean | **10.0%** | query paraphrases miss substring matching |
| `laap_vault` (SQLite, LIKE) | clean | **10.0%** | identical — bottleneck is the matcher, not storage |
| `laap_vault` (SQLite, LIKE, production 6.5k rows) | live | **25.0%** | production noise actually *helps* exact-ish phrasing hits |
| `hybrid` (LIKE + char n-gram TF-IDF) | clean | **100.0%** | same protocol, same data, different matcher — 2026-08-04 |
| `laap_vector` (hash-embedding cosine) | clean | **100.0%** | n-gram hashing to 384-dim, no model download |
| `laap_hierarchy` (L1-L4 tag recall) | clean | **5.0%** | tag-based retrieval is weakest for natural-language queries — honest exposure of its design assumption |
| `laap_qlam` (quantum superposition) | clean | **50.0%** | quantum-state correlation, concept demo |

> **LAAP ecosystem adapters** ship in `recall_bench/backends/`: `vector_backend`, `hierarchy_backend`, `quantum_backend`, plus two protocol layers — `RelationBackend` (knowledge-graph triples) and `SelfModelBackend` (identity/needs/free-energy). The same ruler now measures the memory layer, and the protocols unify what was previously four fragmented systems.

Interpretation: pure substring/LIKE matching retrieves roughly 1-in-4 to 1-in-10 of what a user would naturally ask for. The same data and protocol, measured through a hybrid matcher (LIKE + character n-gram TF-IDF, zero new dependencies), reaches perfect recall on the isolated probe set — confirming the bottleneck is the matching mechanism, not the storage. The hybrid backend ships in this package (`--backend hybrid`) so you can A/B it against your own store. Sample reports: [`examples/results/`](examples/results/).

## Roadmap

- [ ] v0.2 — configurable `--retriever` modes (vector, hybrid) for A/B comparison
- [ ] v0.3 — multi-turn recall (store now, query after N subsequent turns)
- [ ] v0.4 — temporal decay (does recall degrade over simulated time?)
- [ ] v0.5 — cross-backend leaderboard (community-run, published monthly)

## License

Apache-2.0. See [LICENSE](LICENSE).

Part of the [LAAP](https://github.com/liliMozi/LAAP) ecosystem. The benchmark methodology is open — the memories it measures are yours.
