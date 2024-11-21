Here is a collection of notes I got while building this project. 

# General Notes 

* I first started by ordering all my orders by (Price, Time) but I noticed that at certain points. We have multiple orders from the exchange at the exact same timestamp. This would mean that if two orders of the same price had the same time. It could be possible that the bisection algorithm would select the wrong order so I created a new field in my OrderParameters class `process_id` to use instead of time since all the messages are ordered by time, this should maintain the original intention and order. 

* I've spent some time trying to find orders by reference using O(log(n)) time but the `bisect.bisect` algorithm was not playing well with the data structure I had built. The idea was to first bisect by price, then bisect by process_id but the built-in `bisect.bisect` function would not recover the right order leading to multiple errors. The `bisect.insort_left` had no issues however. A customer bisection function could resolve this but in the interest of time at the time of writing, and because most of the search by reference during trades are at the top of the array, I figured the performance wouldn't be too affected. However, during the `remove` or `replace` calls, this could be a source of slowing down performance. Fortunately, that's not the bulk of the messages. 

* Unfortunately, it doesn't seem at times that the messages from the exchange agree with the order book that I am building which lead to certain errors. I have written down below certain cases that would initially produce errors, and I have decided to log those errors when they happen. 

* I've also noticed later that certain trade seems to be happening not at the top of the order book. I've modified my code to select the trade in the order book when the reference in the trade message maps to an existing order in the order book. Otherwise, I would match the trade with the top of the order book. 

* Unfortunately, Python doesn't have pointers which would have made removing orders from the order book quicker by accessing the hash_map(Dictionary of IDs) but Python populates the dictionary with the object so I need to iterate through the order book to remove them. This would be faster using a map object and dictionary in C++. 

* I got some UserWarnings when converting the exchange time into a python time object so usually it would be better to handle those by using a context manager `with warnings.catch_warnings(action="ignore", type=UserWarnings):` but calling this on each line of the dataframe was slowing down the performance significantly so I chose to put it at the header of the script. 

# Error Cases  

## Case 1: No Liquidity Available 

While processing the following order 

Order(Price=14.85, Side=OrderSide.NONE, MessageType=MessageType.TRADE, Size=100, Time=13:25:00:093954, Ref=0, OldRef=0)

We have a message from the exchange that a Trade happened at a Price=14.85

We have the following order at the top of the ASK Book:

Order(Price=14.9, Side=OrderSide.SELL, MessageType=MessageType.ADD, Size=25, Time=12:29:43:044759, Ref=10066878, OldRef=0)

And the following order at the top of the BID Book: 

Order(Price=14.79, Side=OrderSide.BUY, MessageType=MessageType.ADD, Size=100, Time=13:24:51:629095, Ref=17636154, OldRef=0)

So there should be no liquidity available to process that trade. Here is the first 5 level of the level book: 

  Buy Qty  Price Sell Qty
0    None  15.50     1000
1    None  15.30     1631
2    None  15.25       11
3    None  15.08      500
4    None  14.90       25
5     100  14.79     None
6       1  14.76     None
7       1  14.73     None
8      10  14.70     None
9    2530  14.60     None

### Conclusion 

Since there is no liquidity available. I process the trade as being erroneous and log it into the trade_error log file and proceed on. 


## Case 2: Order already processed 

While processing this following Order

Order(Price=14.82, Side=OrderSide.NONE, MessageType=MessageType.TRADE, Size=184, Time=13:55:25:001299, Ref=0, OldRef=0)

We have a TRADE at a Price=14.82 happening. 

The following order is at the top of the BID Book:

Order(Price=14.82, Side=OrderSide.BUY, MessageType=MessageType.ADD, Size=100, Time=13:55:25:001144, Ref=99371134, OldRef=0)

The following order at the top of the BID Book is:

Order(Price=14.82, Side=OrderSide.BUY, MessageType=MessageType.REPLACE, Size=100, Time=13:55:25:001159, Ref=99371162, OldRef=99370994)

Since timestamp of the next order is after we process the first order for the first 100 and following order next 84. 

Later we have the following trade message:

Order(Price=14.82, Side=OrderSide.BUY, MessageType=MessageType.TRADE, Size=100, Time=13:55:25:002830, Ref=99371134, OldRef=0)

The ID of this trade matches the ID of the order that was processed above. 

### Conclusion:

For this reason, I added a part of code that handle invalid trade and proceed from using top of the book log it in trade_error_log log file and proceed on. 


## Case 3: Order Replace have same reference and old reference 

I found that multiple orders such as 

Order(Price=14.77, Side=OrderSide.SELL, MessageType=MessageType.REPLACE, Size=700, Time=13:30:28:977653, Ref=23896366, OldRef=23896366)

Order(Price=14.77, Side=OrderSide.SELL, MessageType=MessageType.REPLACE, Size=700, Time=13:30:29:035695, Ref=23899678, OldRef=23899678)

Have the same new and old references, not sure what I meant to do here so I ignore those messages. 

### Conclusion: 

Added a piece of code to ignore and log these messages. 