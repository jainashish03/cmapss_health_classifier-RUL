# Data

Place the NASA C-MAPSS files from Kaggle here (`data/raw/`):

```
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```
The loader also searches sub-folders (e.g. `data/raw/CMaps/`). The `.txt` files have 26 whitespace-separated
columns without header: unit, cycle, 3 operational settings, 21 sensors.

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|--------|--------------:|-------------:|---------------------:|-------------|
| FD001  | 100 | 100 | 1 | HPC degradation |
| FD002  | 260 | 259 | 6 | HPC degradation |
| FD003  | 100 | 100 | 1 | HPC + Fan degradation |
| FD004  | 249 | 248 | 6 | HPC + Fan degradation |

(The original readme lists the FD004 train/test counts the other way round; the files contain 249 train / 248 test.)
