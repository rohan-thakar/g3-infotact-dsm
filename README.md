# Contextual Predictive Maintenance (IoT Edge AI)

## Internship Project
Infotact Solutions — DSML Internship

---

# Project Overview

This project focuses on building an advanced Predictive Maintenance system using:

- IoT telemetry data
- Contextual external signals
- Time-series feature engineering
- Imbalanced machine learning
- LightGBM classification

The goal is to predict machine failures before they happen.

---

# Week 1 — Day 1 Progress

## Completed Tasks

### Repository Setup
- Initialized GitHub repository
- Created individual working branch
- Added project structure
- Configured .gitignore
- Added requirements.txt

### Project Planning
- Reviewed official execution handbook
- Understood Week 1 objectives
- Planned Kaggle workflow
- Prepared GitHub tracking structure

### Dataset Planning
Selected dataset:
- AI4I Predictive Maintenance Dataset

Planned features:
- Rolling mean
- Rolling standard deviation
- Signal variance
- Feature engineering

---

## Week 1 — Day 2 Progress

### Completed
- Exploratory Data Analysis
- Correlation heatmap
- Feature distributions
- Advanced rolling window features
- Statistical feature engineering
- Data cleaning improvements

### Added Features
- Rolling mean
- Rolling std
- Rolling min/max
- Variance-based analysis

## Week 1 — Day 3 Progress

### Completed

- Built baseline Random Forest model
- Performed train-test split
- Generated classification report
- Created confusion matrix
- Analyzed feature importance

### New Features

- Temperature difference
- Power proxy feature

### Outcome

Successfully trained first predictive maintenance model and evaluated performance.

## Week 1 — Day 4 Progress

### Completed

- Studied class imbalance
- Applied SMOTE oversampling
- Trained improved Random Forest model
- Compared model performance
- Evaluated confusion matrix

### Outcome

Improved handling of rare machine failure cases through data balancing techniques.

## Week 1 — Day 5 Progress

### Completed

- Simulated external contextual data
- Ambient temperature generation
- Factory load simulation
- Humidity simulation
- Contextual feature engineering

### Ablation Study

Compared:

1. Internal telemetry only
2. Telemetry + contextual signals

### Outcome

Verified impact of contextual variables on predictive maintenance performance.

# Week 1 Objectives

- IoT telemetry ingestion
- Signal processing
- Rolling window feature engineering
- Exploratory data analysis
- Baseline preprocessing pipeline

---

# Tech Stack

| Tool | Purpose |
|------|----------|
| Python | Development |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Matplotlib | Visualization |
| Seaborn | Visualization |
| LightGBM | Modeling |
| Kaggle | Training Environment |
| GitHub | Version Control |

---

# Planned Weekly Roadmap

## Week 1
- Dataset ingestion
- Rolling statistics
- Signal processing

## Week 2
- Contextual data fusion
- External feature integration

## Week 3
- LightGBM + SMOTE
- Stratified cross-validation

## Week 4
- Noise sensitivity analysis
- Threshold tuning
- Evaluation dashboard

---

# GitHub Workflow

- Daily commits
- Semantic commit messages
- Individual branch development
- Weekly progress tracking

---
