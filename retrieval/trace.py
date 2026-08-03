"""
retrieval/trace.py — a recording of every decision one retrieval made.

# WHY does this object exist?
# It is what makes the demo look like a product instead of a chatbot. A buyer
# cannot audit a number they never see; a trace that says "40 candidates in, 12
# survived the floor, 5 returned, and here is what was dropped and why" turns the
# abstention behaviour — the actual selling point — into something visible.
#
# # WHY is it a passive RECORDER rather than part of the retriever?
# Because stage 06 must stay the single source of truth for retrieval, and a
# component that both decides and reports will eventually report what it wishes
# had happened. Trace records; it never influences. Nothing here can change a
# ranking, which is why adding it is a zero-behaviour-change operation.

Phase 0 ships the structure. Phase 1 wires `Retriever` to emit into it.

    trace = RetrievalTrace(query="how does metformin work?")
    with trace.stage("ground", note="entity vocabulary") as s:
        s.candidates_in = 14955
        s.candidates_out = 102
        s.drop("wrong drug", 14853)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageRecord:
    """One stage of one retrieval."""
    name: str
    note: str = ""
    ms: float = 0.0
    candidates_in: Optional[int] = None
    candidates_out: Optional[int] = None
    #: reason → how many candidates it removed, e.g. {"below floor": 28}
    dropped: dict = field(default_factory=dict)
    #: the best few scores at this point, for the UI's sparkline
    top_scores: list = field(default_factory=list)

    def drop(self, reason: str, n: int = 1) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + int(n)

    @property
    def kept(self) -> Optional[int]:
        if self.candidates_in is None or self.candidates_out is None:
            return None
        return self.candidates_out

    def summary(self) -> str:
        """One line the UI can render beside the stage name."""
        if self.candidates_in is None or self.candidates_out is None:
            return self.note
        line = f"kept {self.candidates_out:,} of {self.candidates_in:,}"
        if self.dropped:
            why = ", ".join(f"{n:,} {reason}" for reason, n in self.dropped.items())
            line += f" — dropped {why}"
        return line


@dataclass
class RetrievalTrace:
    """Every stage of one retrieval, in order, with timings and drop reasons."""
    query: str
    rewritten: str = ""
    stages: list = field(default_factory=list)
    status: str = ""

    @contextmanager
    def stage(self, name: str, note: str = ""):
        """Record a stage, timing it automatically."""
        record = StageRecord(name=name, note=note)
        t0 = time.perf_counter()
        try:
            yield record
        finally:
            record.ms = (time.perf_counter() - t0) * 1000.0
            self.stages.append(record)

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.stages)

    def as_dict(self) -> dict:
        """JSON-serialisable form — for the audit log and the Retrieval Lab export."""
        return {
            "query": self.query,
            "rewritten": self.rewritten,
            "status": self.status,
            "total_ms": round(self.total_ms, 2),
            "stages": [
                {"name": s.name, "note": s.note, "ms": round(s.ms, 2),
                 "in": s.candidates_in, "out": s.candidates_out,
                 "dropped": s.dropped, "top_scores": [round(x, 4) for x in s.top_scores]}
                for s in self.stages
            ],
        }

    def render_lines(self) -> list:
        """(name, ms, summary) triples — what the pipeline stepper draws."""
        return [(s.name, round(s.ms, 1), s.summary()) for s in self.stages]
