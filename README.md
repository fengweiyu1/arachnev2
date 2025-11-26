# Arachne：面向深度神经网络的搜索式修复

Arachne 是论文 [《Search Based Repair of Deep Neural Networks》](https://arxiv.org/abs/1912.12463) 中提出的深度神经网络搜索式修复框架，通过对神经权重进行故障定位并应用定向补丁来纠正模型错误，同时尽量保持正确预测。本仓库还包含论文中的权重自适应基线 **Apricot**，以及复现七个研究问题（RQ1–RQ7）的脚本和实验流程。

## 仓库结构
- `arachne/`：Arachne 的实现，以及针对 RQ1–RQ7 的脚本、补丁应用工具与定位辅助脚本。
- `apricot/`：Apricot 基线的实现，包含模型定义、训练与修复脚本。
- `LICENSE`：Apache 2.0 许可证。

## 环境与依赖
- Python 3.7 及以上。
- PyTorch 与 torchvision（建议使用 GPU 以加速实验）。
- 常用第三方库：`pandas`、`numpy`、`scikit-learn`、`tqdm`、`Pillow`（来自 torchvision）。
- 推荐的快速安装方式：
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install torch torchvision pandas numpy scikit-learn tqdm pillow
  ```

## 数据与预训练模型
Arachne 的实验默认使用 `final_data/` 目录存放预训练模型与数据集。

1. 预训练模型下载：
   ```bash
   curl -L "https://www.dropbox.com/s/p5n91wtc7b4s6b8/models.tar.gz?dl=0" -o models.tar.gz
   tar -zxvf models.tar.gz -C final_data
   ```
2. 数据集下载（含 GTSRB、LFW、Twitter US Airline Sentiment 以及索引文件）：
   ```bash
   curl -L "https://www.dropbox.com/s/i7ttm3iqyur2usk/data.tar.gz?dl=0" -o data.tar.gz
   tar -zxvf data.tar.gz -C final_data
   ```
3. FashionMNIST 与 CIFAR-10 会在首次通过 torchvision 访问时自动下载。

解压后的目录结构示例：
```
final_data/
  data/            # 数据集与索引文件
  models/          # 论文中使用的预训练模型
```

## 运行 Arachne 实验（RQ1–RQ7）
每个研究问题在 `arachne/` 下都有独立脚本。所有命令默认在仓库根目录执行，并将数据目录（例如 `final_data/data`）作为第一个参数。

### RQ1：故障定位策略比较
```bash
./arachne/rq1.sh <datadir> <loc_method> <which_data> <dest>
```
- `loc_method`：`localiser`（双向定位，BL）、`gradient_loss` 或 `random`。
- `which_data`：`cifar10`、`fashion_mnist` 或 `GTSRB`。
- 示例：`./arachne/rq1.sh final_data/data localiser fashion_mnist results/rq1`

### RQ2：不同定位方法下的修复效果
```bash
./arachne/rq2.sh <datadir> <loc_method> <which_data> <dest>
```
- 补丁保存到 `<dest>`，定位结果保存在 `<dest>/loc`。
- 示例：`./arachne/rq2.sh final_data/data localiser fashion_mnist results/rq2`

### RQ3：针对最常见故障类型的修复
```bash
./arachne/rq3.sh <datadir> <which_data> <dest> <top_n>
```
- `top_n` 表示选择出现频率第 N 高的故障类型（0 为最常见）。
- 示例：`./arachne/rq3.sh final_data/data fashion_mnist results/rq3 0`

### RQ4：修复与保持性能的权衡
```bash
./arachne/rq4.sh <datadir> <which_data> <dest> <patch_aggr>
```
- `patch_aggr` 对应 alpha 超参数，用于控制修复与保持的权衡（默认 `10`）。
- 示例：`./arachne/rq4.sh final_data/data fashion_mnist results/rq4 10`

### RQ5：模型多样性评估
```bash
./arachne/rq5.sh <datadir> <which> <dest>
```
- `which`：`cnn1`、`cnn2`、`cnn3`、`GTSRB` 或 `fm`（FashionMNIST）。
- 示例：`./arachne/rq5.sh final_data cnn1 results/rq5`

### RQ6：LFW 性别分类场景
```bash
./arachne/rq6.sh <datadir> <dest>
```
- 修复最常见的错误类型（female → male）。
- 示例：`./arachne/rq6.sh final_data/data results/rq6`

### RQ7：Twitter 美国航空情感分析
```bash
./arachne/rq7.sh <datadir> <dest>
```
- 修复最常见的错误类型（neutral → negative）。
- 示例：`./arachne/rq7.sh final_data/data results/rq7`

## 评估已修复模型
使用 `run_mdl.sh` 应用生成的补丁并输出预测：
```bash
./arachne/run_mdl.sh <rq> <datadir> <which_data> <path_to_patch> <top_n> [which]
```
- `rq`：研究问题编号（1–7）。
- `which_data`：`cifar10`、`fashion_mnist`、`GTSRB`、`fm_for_rq5`、`lfw` 或 `us_airline`；RQ5–RQ7 可留空。
- `path_to_patch`：对应 RQ 脚本生成的补丁文件路径。
- `top_n`：与生成补丁时使用的值一致（用于 RQ3/RQ4）。
- `which`：仅 RQ5 需要指定模型类型（如 `cnn1`）。
- 示例：`./arachne/run_mdl.sh 5 final_data/data cifar10 results/rq5/model.misclf-rq5.0.0-3-5.pkl 0 cnn1`

### 输出文件
- 补丁文件：`model.misclf-rq#.<seed>-<true_label>-<pred_label>.pkl`（RQ3–RQ7）或 `model.rq2.<seed>.pkl`（RQ2）。
- 预测结果：保存在 `<dest>/pred` 下的 CSV，列包含 `true`、`pred`、`new_pred` 与 `init_flag`。

## Apricot 基线
Apricot 提供基线权重自适应方法。

1. 准备中间模型与修复模型（iDLM 和 rDLM）：
   ```bash
   mkdir -p apricot/weights apricot/rDLM_weights
   python apricot/iDLM_train.py
   python apricot/rDLM_train.py
   ```
   如需调整网络结构，可修改 `apricot/iDLM_train.py` 与 `apricot/rDLM_train.py` 中的 CNN 配置。
2. 运行 Apricot 进行修复（支持使用 `-err_src` 与 `-err_dst` 指定定向模式）：
   ```bash
   python apricot/weight_adjust.py -err_src 3 -err_dst 5
   ```

## 引用
若在学术研究中使用本代码，请引用：
```
@misc{sohn2019search,
      title={Search Based Repair of Deep Neural Networks},
      author={Jeongju Sohn and Sungmin Kang and Shin Yoo},
      year={2019},
      eprint={1912.12463},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
}
```
