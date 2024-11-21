## About The Project

This project was done in the candidate interview process for Quant Trade role at Old Mission Capital (2024). 
The goal is to generate an order book by processing messages from the exchange and display several information every 30 minutes starting at 13:30 until 20:00. 

* Top 5 Level of the Order Book 
* VWAP
* Cumulative Volume 

## Getting Started

Project Structure 

├── build                   # Compiled files 
├── docs                    # Documentation files 
├── src                     # Source files 
├── tests                   # Automated tests (empty)
├── logs                    # Logs of Order Book
├── LICENSE
├── NOTES.md                # Personal Notes related to some cases with examples
├── pyproject.toml          # Instructions for pip to install the dependencies
└── README.md               # This document 


Code Structure under src/omc_order_book 


├── loggings_configs                # JSON configs of the different loggers 
├── data                            # Data Folder 
├── utils                           # Directory of useful objects for OrderBook
│   ├── order_book_dict.py          # ChainMap Dictionary for merging ask dict with bid dict
│   ├── order_info.py               # All information about creating order objects 
│   └── sorted_default_dict.py      # Dictionary with self sorted keys for Level Book 
├── \_\_init\_\_.py                     # Used to initiate all the loggers 
├── _utils__.py                     # Used to import all the objects created under utils
├── order_book.py                   # OrderBook object 
└── main.py                         # Main script to run the different testers 


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
4. Install omc_order_book package
   ```sh
   pip install . 
   ```

## Usage

To run the main script that will generate the output required for the deliverable. Simply run from the terminal.

```
sh 
project_main
```
It should output all the required output asked. 

Alternatively, you can also see all the logs in after running the script.

`<source folder>/logs/deliverable_log.log`



To run a second script that generates more logs of the trades and errors. Run 

```
sh
process_exchange_messages --log
```

This will not output the deliverable information but will log a lot of different errors and information to the different log files.

## Data Structures Used 

1. Dictionary of ID --> Order for fast access to Order information when needed. 
2. Array of sorted orders for the ASK orders
   2.1. Insertion: O(log(n))
   2.2. Access: O(n)
3. Array of reversed sorted orders for the BID orders
   3.1. Insertion: O(log(n))
   3.2. Access: O(n)
4. Sorted Dictionaries of price levels --> size for quick access to top price levels. 
   4.1. The idea with this was to create a defaultdictionary of integers to quickly increment the size without manually creating a key. 
   4.2. Deleting the key when the size is 0 to keep the sorted keys only for price where there is liquidity. 
   4.3. Quick access to the top prices levels by either looking at the start of the keys or the end whether ask/bid. 
5. Order Class with defined dunder methods. 
   5.1. Here I wanted to make the code look more pythonic by comparing orders when using bisectional method when inserting orders or comparing two references.   

## Roadmap

- [ ] Access orders in order book in O(log(n)) time when removing by id. 
- [ ] Figure out how to point dictionary key to the object (similar to C++ pointers) for faster access in order book if possible. 
- [ ] Add tester for the different objects.  


## Additional 

Please read the NODES.md file for more information about certain errors I've observed and logged as well as some observations throughout the project. 

If there are any questions or problems running the code, please feel free to contact me. Thanks.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


## Contact

Your Name - Laurent Lanteigne - laurent.lanteigne@gmail.com
