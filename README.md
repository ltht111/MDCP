# MDCP

# SuStaIn Stage 测试说明

该目录用于使用已经训练完成的 SuStaIn 模型，预测新受试者所处的 SuStaIn Stage。

## 文件说明

### `sustain.py`

SuStaIn 测试脚本，主要完成以下操作：

1. 读取测试数据 `test.csv`。
2. 检查测试数据是否包含模型需要的 10 个影像特征。
3. 读取训练完成的 `result_new_norm_subtype0.pickle` 模型。
4. 检查模型事件数量是否与当前的 `Z_vals` 设置一致。
5. 使用模型的 MCMC 后验样本预测每位受试者的 SuStaIn subtype 和 Stage。

脚本中的 `Z_MAX` 和 `Z_VALS` 必须与训练模型时的设置完全一致，不能单独修改。

### `test.csv`

需要预测的新受试者数据。每一行代表一位受试者，每一列代表一个影像特征。

必须包含以下 10 列，列名需要保持一致：

```text
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

脚本会按照上述固定顺序读取特征，因此 CSV 中存在其他信息列不会影响预测。

### `result_new_norm_subtype0.pickle`

训练完成的 SuStaIn 模型文件。

- `subtype0.pickle` 表示包含 1 个 subtype 的模型。


## 运行方法
```
```bash
python sustain.py \
  --data test.csv \
  --model result_new_norm_subtype0.pickle \
  --output-dir xxx
```

需要测试其他数据时，可以指定数据、模型和输出目录：

```bash
python sustain.py \
  --data other_test.csv \
  --model result_new_norm_subtype0.pickle \
  --output-dir xxx
```
