"""QAService.ask: the single QA entry point for API, CLI, evaluation and MCP (SDD 5.2)."""
from __future__ import annotations

from lawrag.models.qa import AskOptions, ContextPack, Draft, QAResponse, QueryAnalysis
from lawrag.qa.workflow import Workflow
from lawrag.services.container import Deps
from lawrag.verify.deterministic import deterministic_checks


class QAService:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.workflow = Workflow(deps)

    def ask(self, question: str, options: AskOptions | None = None, run_id: str | None = None) -> QAResponse:
        return self.workflow.ask(question, options, run_id)

    def run_state(self, question: str, options: AskOptions | None = None, run_id: str | None = None) -> dict:
        return self.workflow.run(question, options, run_id)

    def verify(self, answer: str, context_uids: list[str], question: str | None = None):
        """POST /verify: D-checks on arbitrary text against a context made of the given uids."""
        from lawrag.legal import uid as U
        from lawrag.retrieval.context_builder import mapping_block, provision_block
        from lawrag.retrieval.mapping_facts import to_fact
        store = self.deps.store
        blocks, facts = [], []
        for u in context_uids:
            if U.MAPPING_ID_RE.match(u):
                m = store.get_mapping(u)
                if m:
                    f = to_fact(m, store)
                    facts.append(f)
                    blocks.append(mapping_block(f))
                continue
            p = store.get_provision(u)
            provs = [p] if p else store.provisions_of(u)
            for pp in provs:
                blocks.append(provision_block(pp, store.get_section(pp.section_uid), qualifier=pp.is_qualifier))
        ctx = ContextPack(blocks=blocks, mapping_facts=facts)
        analysis = self.deps.analyzer.analyze(question) if question else QueryAnalysis()
        return deterministic_checks(Draft(answer=answer), ctx, analysis, store, self.deps.commencement,
                                    list(self.deps.cfg.acts))
