# WAC — World AgroCommodities: Earth Observation for Deforestation-Free Supply Chains

This repository contains code, models, and utilities developed under the **World AgroCommodities (WAC)** project, which aims to support deforestation-free supply chains (EUDR) using satellite data and ancillary sources.

## Project Goals

- Map commodity parcels & crop types over demonstration sites
- Detect deforestation / land conversion after December 2020
- Provide scalable, open, reproducible methods for EU Member States by making use of **openEO** and **CDSE**
- Validate algorithms and support national uptake

## Repository Structure

Below is an example structure. You should adjust to match what you currently have. If you are interesting in testing out the commodity mapping; *classification* is the entry point

```
WAC/
├── classification/         # Scripts for performing commodity classification tasks
├── in_situ/                # [experimental] scripts for onboarding in-situ training data
├── model_training/         # scripts detailing the model architecture definitions, training scripts, utilities
├── requirements.txt        # Python dependencies
```

## Installation & Dependencies


Clone the repository:
```git clone https://github.com/masolele/WAC
cd WAC
```


(Optional) Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

Ensure **openEO** is installed
