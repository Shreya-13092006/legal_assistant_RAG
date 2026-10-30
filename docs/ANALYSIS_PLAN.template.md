# Analysis plan (copy to docs/ANALYSIS_PLAN.md and freeze BEFORE running the test split)

The runner refuses `--split test` until `docs/ANALYSIS_PLAN.md` exists; its sha256 is logged with every test run.

1. Configurations: E1 R-V, R-B, R-G, R-H, R-H−G; E2 V0, V-D, V-DJ; E3 T0, T1 (SDD 7.3). Retrieval config for E2/E3: R-H
   (or the dev-chosen alternative, recorded here with the reason).
2. Primary comparisons: E1 R-H vs R-V on Recall@8 (Wilcoxon + paired bootstrap CI); E2 V-DJ vs V0 on deterministic
   hallucination rate (exact McNemar + CI); E3 T1 vs T0 on acceptable outcome (exact McNemar + CI).
3. Secondary comparisons (Holm within each experiment): SDD 7.4.
4. τ_partial, evidence-gate settings, graph caps and prompts frozen at: <values from dev>.
5. Models: generator <...>, judge <...>, eval judge <...>; embedding <...> (from the M3 benchmark).
6. Simulation-based power estimate from dev discordance rates: <...>.
7. Exclusions: items fixed or removed for DATA errors are listed with reasons.
