# Identifying Confabulation Hotspots of Large Language Models in Radiology through Semantic Entropy 

## Introduction
This repository contains the code used for the (not yet published) paper titled _"[TODO ADD PAPER]"_. If you find this code or research valuable for your work, please consider citing our paper: [TODO ADD PAPER] 

We have created a set of 90 questions, which span 4 different areas:
- Guidelines and Indications
- Image Acquisition
- Imaging Education
- Research

The CSV files containing all the questions can be found in the [`questions`](questions/) folder.

## Requirements
Note: We used Ubuntu 22.04
1. Ensure `conda` is installed on your system.
2. Set up the conda environment using the provided [`environment.yml`](environment.yml) file:
```
$ conda env create --name semantic_entropy_radiology --file=environment.yml
```
3. Activate the environment:
```
$ conda activate semantic_entropy_radiology
```
The environment setup is now complete.

## Quick Evaluation
To reproduce our results, run:
```
$ python3 eval.py
```
This command processes the cached LLM responses (stored in `cache`) to regenerate our findings. For a complete evaluation including new LLM prompts, see the "Reproducing the Results" section below.

## Reproducing the Results
Note: Due to the non-deterministic nature of LLMs, new prompting may yield slightly different results.

1. Configure credentials:
   - Open [`CONFIG.py`](CONFIG.py)
   - Update `LOCAL_KEY`, `LOCAL_ENDPOINT`, `OPENAI_KEY` and `AZURE_ENDPOINT` with your credentials.

2. Prepare the environment:
   - Clear the `cache` directory to ensure a clean evaluation.

3. Generate new answers:
```
$ python3 generateAnswers.py
```
Results are saved in the `cache` directory with filenames prefixed with `EVAL`.

4. Run the evaluation:
```
$ python3 eval.py
```
This step utilizes GPT4o for semantic clustering and answer comparison. Previous entailment results are cached in [`entailmentCacheFile_GPT4o.csv`](cache/entailmentCacheFile_GPT4o.csv).

## Project Components

### Configuration Files
- [`CONFIG.py`](CONFIG.py): Contains the configurations needed to run the code locally.

### Dataset Structure
- [`RadDataset.py`](RadDataset.py): Defines the dataset structure used in the project.

### Core Functionality
#### Answer Generation
- [`generateAnswers.py`](generateAnswers.py): Handles answer generation from the dataset
  - Generates primary answers (ID `0`) with temperature `0.1` for accuracy assessment
  - Creates additional answers for semantic entropy computation
  - Stores results in designated output files

#### LLM Integration
- [`promptLLM.py`](promptLLM.py): Implements the logic for prompting various large language models (LLMs).

#### Semantic Entropy Calculation
- [`clusterAnswers.py`](clusterAnswers.py): Implements semantic entropy calculations and evaluation metrics (AUROC, AURAC)
- [`EntailmentCheck.py`](EntailmentCheck.py): Manages clustering and entailment verification
  - Implements prompt caching for efficient repeated executions

#### Radiologist correction
To verify answer correctness, all correctness labels were reviewed by a board-certified radiologist. This led to changes in some reported values. The corresponding corrections are included in [`entailmentCacheFile_GPT4o.csv`](cache/entailmentCacheFile_GPT4o.csv).

#### Evaluation Tools
- [`eval.py`](eval.py): Handles the evaluation pipeline
- Features bootstrapping analysis for confidence interval calculations across semantic entropy thresholds