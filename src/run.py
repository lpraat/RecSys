import random

from src.const import NUM_TRACKS
from src.data import Cache
from src.metrics import leave_one_out

cache = Cache()

# Just testing with some random generated data
preds       = []
test_set    = []

for i in range(1000):
    pred = []

    # Generate random predictions  
    for i in range(10):
        pred.append(random.randint(0, 10))
    preds.append(pred)

    # Generate random test entry
    test_set.append(random.randint(0, 10))

print(leave_one_out(preds, test_set))