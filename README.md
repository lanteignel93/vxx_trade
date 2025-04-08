# About The Project

The goal of this project is yet to be fully defined. I'm using this project for the following mean reasons 

* Software engineering practice 
   * Packaging a Python Project 
   * Building Modular OOP Trading System 
   * Investigating and learning best practices 
   * Learning design patterns 
* Trading 
   * Building a research platform 
   * Learning data techniques 
   * Build a backtest 
   * Optimizing trading strategy position under risk constraints 

# Getting Started

## Project Structure 

├── build                   &emsp; &emsp; &emsp; # Compiled files   
├── docs                    &emsp; &emsp; &emsp; # Documentation files    
├── src                     &emsp; &emsp; &emsp; # Source files    
├── tests                   &emsp; &emsp; &emsp; # Automated tests (empty)   
├── logs                    &emsp; &emsp; &emsp; # Logs of the project    
├── LICENSE    
├── NOTES.md                &emsp; &emsp; &emsp; # Personal Notes related to some cases with examples   
├── .gitignore              &emsp; &emsp; &emsp; # List of files to be ignored    
├── .gitattributes              &emsp; &emsp; &emsp; # List of github related attributes    
├── .pre-commit-config.yaml &emsp; &emsp; &emsp; # Configs to manage the commits        
├── requirements.txt        &emsp; &emsp; &emsp; # List of libraries versions of the latest stable built    
├── pyproject.toml          &emsp; &emsp; &emsp; # Instructions for pip to install the dependencies and general packaging    
└── README.md               &emsp; &emsp; &emsp; # This document 


## Code Structure under src/vxx_trade

TODO: Explain the different python scripts under vxx_trade/


# Package Maintenance 
## Prerequisites

Python >= 3.10.12

## Installation

Ideally installing this on linux would be better, this was only tested on linux Ubuntu (22.04). 

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
