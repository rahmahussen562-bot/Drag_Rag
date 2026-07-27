"""
rxnorm.py — optional runtime drug-name normalization via the NLM RxNorm API.

Purpose: map brand names / misspellings the offline resolver misses (e.g.
"Tylenol" → "acetaminophen") to a standard ingredient name, so InteractionDB can
then find it. This is a *network* dependency (rxnav.nlm.nih.gov) and entirely
optional — every method degrades gracefully to None so the app works offline.

NOTE: the RxNorm *Interaction* API was discontinued (Jan 2024); only the
normalization endpoints used here remain live. Interactions come from DDInter.
"""
from __future__ import annotations

from typing import Optional

RXNAV = "https://rxnav.nlm.nih.gov/REST"


class RxNormResolver:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self._cache: dict = {}

    def available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{RXNAV}/version.json", timeout=self.timeout)
            return r.ok
        except Exception:
            return False

    def standard_name(self, name: str) -> Optional[str]:
        """Best-effort standard ingredient name for `name`, or None."""
        if not name or not name.strip():
            return None
        key = name.strip().lower()
        if key in self._cache:
            return self._cache[key]
        out = None
        try:
            rxcui = self._approx_rxcui(name)
            if rxcui:
                out = self._ingredient_name(rxcui) or self._rxnorm_name(rxcui)
        except Exception:
            out = None
        self._cache[key] = out
        return out

    # ---- internals ---------------------------------------------------------
    def _approx_rxcui(self, name: str) -> Optional[str]:
        import requests
        r = requests.get(f"{RXNAV}/approximateTerm.json",
                         params={"term": name, "maxEntries": 1}, timeout=self.timeout)
        r.raise_for_status()
        cand = r.json().get("approximateGroup", {}).get("candidate", []) or []
        return cand[0].get("rxcui") if cand else None

    def _ingredient_name(self, rxcui: str) -> Optional[str]:
        """Prefer the base ingredient (maps brand → active ingredient)."""
        import requests
        r = requests.get(f"{RXNAV}/rxcui/{rxcui}/related.json",
                         params={"tty": "IN"}, timeout=self.timeout)
        r.raise_for_status()
        groups = r.json().get("relatedGroup", {}).get("conceptGroup", []) or []
        for g in groups:
            props = g.get("conceptProperties") or []
            if props:
                return props[0].get("name")
        return None

    def _rxnorm_name(self, rxcui: str) -> Optional[str]:
        import requests
        r = requests.get(f"{RXNAV}/rxcui/{rxcui}/property.json",
                         params={"propName": "RxNorm Name"}, timeout=self.timeout)
        r.raise_for_status()
        concepts = r.json().get("propConceptGroup", {}).get("propConcept", []) or []
        return concepts[0].get("propValue") if concepts else None
