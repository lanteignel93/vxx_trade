# About The Project

The goal of this project is yet to be fully defined. 


* Top 5 Level of the Order Book 
* VWAP
* Cumulative Volume 

# Getting Started

## Project Structure 

├── build                   # Compiled files 
├── docs                    # Documentation files 
├── src                     # Source files 
├── tests                   # Automated tests (empty)
├── logs                    # Logs of the project
├── LICENSE
├── NOTES.md                # Personal Notes related to some cases with examples
├── .gitignore              # List of files to be ignored
├── .pre-commit-config.yaml # Set of configs to manage the commits of the project, mostly formatting
├── requirements.txt        # List of libraries versions of the latest stable built
├── pyproject.toml          # Instructions for pip to install the dependencies and general packaging
└── README.md               # This document 


## Code Structure under src/vxx_trade

TODO: Explain the different python scripts under vxx_trade/

### Prerequisites

Python >= 3.10.12

### Installation

Ideally installing this on linux would be better, this was only tested on linux. 

1. `cd <source folder>` # should see the project structure as mention above when doing `ls` command. 
2. Create Python venv
   ```sh
   python -m venv venv
   ```
3. Activate virtualenv
   ```sh
   source ./venv/bin/activate
   ```
4. Install vxx_trade
   ```sh
   pip install . 
   ```
