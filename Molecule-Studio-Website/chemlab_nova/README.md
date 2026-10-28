# Molecule Studio — Cheminformatics Virtual Lab

A redesigned, chemistry-inspired web app based on the supplied Cheminformatics Virtual Laboratory project. It uses a dark scientific interface, cyan/blue accents, animated 3D molecular models, and a responsive Streamlit layout.

## Features

- SMILES parsing and validation with RDKit
- 2D molecular structure rendering
- Interactive 3D conformer generation with rotation, zoom and selectable rendering styles
- Optional van der Waals surface
- Molecular descriptors: formula, molecular weight, exact mass, LogP, TPSA, hydrogen-bond donors/acceptors, rotatable bonds, ring counts, fraction Csp3 and molar refractivity
- CSV export and personalized PDF report
- Multi-molecule structure/property comparison with chart and CSV export
- Theory/reference pages and a five-question knowledge check
- Example molecules: ethanol, benzene, phenol, acetic acid, aspirin, caffeine, glucose and lactic acid

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Notes

- RDKit calculates descriptors from the submitted molecular graph.
- The 3D view generates and optimizes a plausible conformer; it is not an experimental structure and molecules may have multiple conformations.
- This is an educational tool, not a substitute for experimental measurements or expert chemical review.
- Google Fonts and the 3D viewer may require an internet connection in the browser.
