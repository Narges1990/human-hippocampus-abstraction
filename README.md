# Neuronal Selectivity and Geometric Alignment in Human Hippocampal Abstract Generalization

Official MATLAB and Python implementation for the paper:
Neuronal selectivity and geometric alignment in the human hippocampus support abstract generalization
Armin Hakkak Moghadam Torbati and Narges Davoudi

## Overview
This repository contains the computational pipeline used to investigate the neural mechanisms underlying abstract generalization in the human hippocampus. The analysis combines a computational toy model with single-unit recordings from human patients to examine how neuronal selectivity profiles, low-dimensional population dynamics, and geometric alignment drive Cross-Condition Generalization Performance (CCGP).

## Key Findings
- Information vs. Abstraction: Simply increasing the proportion of task-selective neurons improves encoding strength but is insufficient to boost cross-context generalization.
- Category-Like Neurons Drive CCGP: Category-like neurons play a specific role in enhancing CCGP compared to identity-like neurons.
- Geometric Alignment: Geometric alignment of category axes across contexts—rather than simple category separation—serves as the main geometric driver for abstract generalization.
- Mediation Mechanism: Mediation analysis shows that category-like neurons promote CCGP primarily by establishing aligned category vectors across different contexts.

## Repository Structure
human-hippocampus-abstraction/
├── LICENSE
├── README.md
├── requirements.txt
├── toy_model/
│   └── toy_model_analysis.py
└── hippocampal_analysis/
    ├── Category_neurons_main_code.m
    ├── Identity_neurons_main_code.m
    ├── build_pseudopopulation_independent.m
    ├── compute_category_CCGP.m
    ├── compute_context_CCGP.m
    ├── compute_condition_means.m
    ├── compute_category_axis_alignment.m
    ├── cosine_similarity.m
    ├── create_independent_trial_split.m
    ├── find_min_K_independent.m
    ├── getPredictorP.m
    ├── sample_neurons.m
    ├── scenario_encoding_summary.m
    └── train_test_decoder.m

## Getting Started

### Prerequisites

#### Python (Toy Model)
Install required Python libraries:
pip install -r requirements.txt

#### MATLAB (Hippocampal Analysis)
- Recommended: MATLAB R2024b or newer.
- Required: Statistics and Machine Learning Toolbox.

## Usage

### 1. Toy Model Simulations
python toy_model/toy_model_analysis.py

### 2. Human Single-Unit Data Pipeline
1. Download the public single-unit dataset from Courellis et al., Nature (2024).
2. Add the downloaded dataset path and the hippocampal_analysis/ directory to your MATLAB path.
3. Run the main analysis scripts in MATLAB:
   run('hippocampal_analysis/Category_neurons_main_code.m');
   run('hippocampal_analysis/Identity_neurons_main_code.m');

## Data Availability
The empirical single-unit neural recordings analyzed in this project were obtained from publicly available medial temporal lobe (MTL) human data made available by Courellis et al. (Nature, 2024).

## Citation & Contact
If you use this codebase or methodology in your research, please cite our paper.

Correspondence:
- Armin Hakkak Moghadam Torbati (armin.hakkak.moghadamtorbati@ulb.be)
- Narges Davoudi (n.davoudi@studenti.unina.it)


