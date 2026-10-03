# AHENK

This repository contains the executable research code and environment definition for the AHENK project.

## Pre-publication code-only release

The manuscript, bibliography, research data, model outputs, result files, figures, tables, provider responses, and review materials are intentionally not distributed in this repository. Their absence is deliberate and does not indicate an incomplete upload.

## Contents

- `scripts/`: data-processing, modeling, evaluation, and validation programs.
- `environment.yml`: Conda environment definition.
- `LICENSE`: license for the repository source code.
- `NOTICE.md`: third-party data and content notice.

## Installation

```bash
conda env create -f environment.yml
conda activate ahenk
```

The scripts expect locally supplied inputs. No source dataset, derived dataset, credential, API response, trained model, prediction, or experimental result is included.

This snapshot intentionally omits private preparation and publication-management components. It is not a complete, standalone reproduction archive. The source code itself exposes algorithmic choices and frozen experimental configuration; excluding study files does not conceal the method.

## Public API status

This is a pre-publication research-code snapshot. Command-line interfaces and file-layout expectations may change before the archival release.

## License

The source code is available under the MIT License. This license does not grant rights to any third-party data that may be used with the code.