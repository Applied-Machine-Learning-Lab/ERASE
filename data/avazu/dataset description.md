## Dataset Description

[Click-Through Rate Prediction | Kaggle](https://www.kaggle.com/competitions/avazu-ctr-prediction/data)

## File descriptions

- **train** - Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies.
- **test** - Test set. 1 day of ads to for testing your model predictions. 
- **sampleSubmission.csv** - Sample submission file in the correct format, corresponds to the All-0.5 Benchmark.

## Data fields

- id: ad identifier
- click: 0/1 for non-click/click
- hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
- C1 -- anonymized categorical variable
- banner_pos
- site_id
- site_domain
- site_category
- app_id
- app_domain
- app_category
- device_id
- device_ip
- device_model
- device_type
- device_conn_type
- C14-C21 -- anonymized categorical variables

## Notes

For the test file does not contain the labels, so we just use the training file.

## read dtypes

```bash
Memory usage of dataframe is 7402.76 MB
Memory usage after optimization is: 1773.58 MB
Decreased by 76.0%
dtypes:  click                int8
hour                int16
C1                   int8
banner_pos           int8
site_id             int16
site_domain         int16
site_category        int8
app_id              int16
app_domain          int16
app_category         int8
device_id           int32
device_ip           int32
device_model        int16
device_type          int8
device_conn_type     int8
C14                 int16
C15                  int8
C16                  int8
C17                 int16
C18                  int8
C19                  int8
C20                 int16
C21                  int8
dtype: object
preprocess avazu done!
```