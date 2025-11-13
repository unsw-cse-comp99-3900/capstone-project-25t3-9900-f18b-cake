## Engineering Practices

### CI/CD & Code Quality
To ensure reproducibility and maintainability, our team followed CI/CD-style workflows.  
We used **GitHub branch protection**, **feature branches linked to Jira tickets**, and **pull requests with code reviews** to simulate continuous integration.  
Each major component (`complete_data_pipeline.py`, `label_generation_pipeline.py`, `training.py`) includes automated logging, retry mechanisms, and runtime validation to guarantee reproducible results.


### Build & Deployment Environment
All scripts were designed to run both locally and on the **Databricks cluster**, allowing consistent builds and isolated execution environments.  
Command-line interfaces (`argparse`) were added to support parameterized execution, simulating a professional deployment pipeline.


### Logging & Monitoring
The data pipelines output structured logs (`.jsonl`) and runtime metrics (`.json`),  
providing visibility for debugging, benchmarking, and tracking speedups from parallel training.  
This mimics **observability practices** used in CI/CD monitoring systems.


### Documentation & Internal API Contracts
Modules are accompanied by configuration and manifest files (`manifest.mf`, `hyperparams.yaml`, `label_mapping.yaml`)  
that define clear input-output contracts, forming a **living backend documentation**.  
These ensure downstream training jobs operate under consistent configurations.



### Jira & Agile Integration
Each code component was linked to corresponding Jira tickets  
(e.g., `F18BCAKE-1-data-isolation-pipeline`, `F18BCAKE-9-job-scheduling`),  
demonstrating traceable version control and parallel team collaboration consistent with agile CI/CD principles.


###  Overall
By combining reproducible pipelines, automated scripts, GitHub-based collaboration,  
and Databricks job scheduling, the team successfully simulated **industry-grade CI/CD engineering practices**  
suitable for a data science project.

##  Project Structure 

capstone-project-25t3-9900-f18b-cake/
├── data/ # Data preprocessing and label generation 
│ ├── complete_data_pipeline.py # Full data isolation and cleaning pipeline 
│ └── label_generation_pipeline.py # Label generation workflow 
│
├── src/ # Source code for model training 
│ ├── configs/ # Configurations & mapping files 
│ │ ├── hyperparams.yaml # Model hyperparameter settings 
│ │ ├── label_mapping.yaml # Label dictionary for class mapping 
│ │ └── manifest.mf # Pipeline manifest 
│ │
│ ├── jobs/ # Databricks job orchestration scripts 
│ │ ├── 01_data_prep.py # Data preparation job 
│ │ ├── 02_train_per_class.py # Per-class model training 
│ │ ├── 03_aggregate_and_visualize.py # Result aggregation & visualization 
│ │ └── manifest.mf # Job manifest file 
│ │
│ └── mch/
│ └── models/
│ └── training.py # Main training entry 
│
├── results/ # Model outputs & evaluation results 
│ └── result-training/
│
├── ENGINEERING_PRACTICES.md # CI/CD & Engineering documentation 
├── README.md # Main project description 
└── requirements.txt # Python dependencies 