# Drug-drug interaction data — attribution

The CSV files in this directory (`code_A.csv` … `code_V.csv`) are from **DDInter 2.0**.

- **Source:** https://ddinter2.scbdd.com  (download page: `/download/`)
- **Citation:** Tian et al. *DDInter 2.0: an enhanced drug interaction resource with expanded
  data coverage, new interaction types, and improved user interface.* Nucleic Acids Research,
  Volume 53, Issue D1, 2025. https://doi.org/10.1093/nar/gkae726
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0). https://creativecommons.org/licenses/by-nc-sa/4.0/

This project uses the data for **non-commercial, educational** purposes and redistributes it
under the same license, with attribution, per CC BY-NC-SA 4.0.

Each row: `DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level` where `Level ∈ {Major, Moderate,
Minor, Unknown}`. Loaded by [`interactions.py`](../../interactions.py).

To refresh: download the 8 category CSVs from https://ddinter2.scbdd.com/static/media/download/
(named `ddinter_downloads_code_{A,B,D,H,L,P,R,V}.csv`) and save them here as `code_{X}.csv`.
