# MDCP

# SuStaIn Stage Test Instructions

This directory is used to predict the SuStaIn Stage of new subjects using a pre-trained SuStaIn model.

## File Description

### `sustain.py`

The SuStaIn test script, which performs the following operations:

1. Reads the test data `test.csv`.

2. Checks if the test data contains the 10 image features required by the model.

3. Reads the pre-trained `result_new_norm_subtype0.pickle` model.

4. Checks if the number of events in the model matches the current `Z_vals` setting.

5. Predicts the SuStaIn subtype and Stage for each subject using the model's MCMC posterior samples.

The `Z_MAX` and `Z_VALS` settings in the script must be exactly the same as those used during model training and cannot be modified individually.

### `test.csv`

Data of new subjects to be predicted. Each row represents a subject, and each column represents an image feature.

The following 10 columns must be included, and column names must be consistent:

``text
ADC_Mean
ADC_Median
ADC_5th
ADC_95th
ADC_Skewness
DCE_MER
DCE_WIR
DCE_WOR
DCE_SER
DCE_iAUC

```
The script will read features in the above fixed order, so the presence of other information columns in the CSV will not affect the prediction.

### `result_new_norm_subtype0.pickle`

The trained SuStaIn model file.

- `subtype0.pickle` indicates a model containing 1 subtype.

## Running Method

```
```bash
python sustain.py \
--data test.csv \
--model result_new_norm_subtype0.pickle \
--output-dir xxx

```

When testing other data, you can specify the data, model, and output directory:

```bash
python sustain.py \
--data other_test.csv \

--model result_new_norm_subtype0.pickle \

--output-dir xxx

```
