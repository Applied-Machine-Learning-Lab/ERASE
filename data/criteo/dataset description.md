        ------ Display Advertising Challenge ------

Dataset: dac-v1

This dataset contains feature values and click feedback for millions of display 
ads. Its purpose is to benchmark algorithms for clickthrough rate (CTR) prediction.
It has been used for the Display Advertising Challenge hosted by Kaggle:
https://www.kaggle.com/c/criteo-display-ad-challenge/

===================================================

Full description:

This dataset contains 2 files:
  train.txt
  test.txt
corresponding to the training and test parts of the data. 

====================================================

Dataset construction:

The training dataset consists of a portion of Criteo's traffic over a period
of 7 days. Each row corresponds to a display ad served by Criteo and the first
column is indicates whether this ad has been clicked or not.
The positive (clicked) and negatives (non-clicked) examples have both been
subsampled (but at different rates) in order to reduce the dataset size.

There are 13 features taking integer values (mostly count features) and 26
categorical features. The values of the categorical features have been hashed
onto 32 bits for anonymization purposes. 
The semantic of these features is undisclosed. Some features may have missing values.

The rows are chronologically ordered.

The test set is computed in the same way as the training set but it 
corresponds to events on the day following the training period. 
The first column (label) has been removed.

====================================================

Format:

The columns are tab separeted with the following schema:
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>

When a value is missing, the field is just empty.
There is no label field in the test set.

====================================================

Dataset assembled by Olivier Chapelle (o.chapelle@criteo.com)
    
```
Memory usage of dataframe is 13989.45 MB
Memory usage after optimization is: 3322.49 MB
Decreased by 76.2%
dtypes:  0      int8
1      int8
2      int8
3      int8
4      int8
5     int16
6     int16
7      int8
8      int8
9      int8
10     int8
11     int8
12     int8
13     int8
14    int16
15    int16
16    int32
17    int32
18    int16
19     int8
20    int16
21    int16
22     int8
23    int32
24    int16
25    int32
26    int16
27     int8
28    int16
29    int32
30     int8
31    int16
32    int16
33     int8
34    int32
35     int8
36     int8
37    int32
38     int8
39    int32
dtype: object
2023-09-06  16:00:58 
save to file...
preprocess criteo done!
```