#!/bin/bash

python3 adapt.py --i 0 --pics 10 --savepath test --source eth --target mpii --gpu 0 --shuffle --js --oma2 --sg
python3 test.py --i -1 --p 0 --savepath test --target mpii