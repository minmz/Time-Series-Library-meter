# Agentic Flow Forecast README

这个 README 只说明本次新增的 `agentic_flow_forecast` 实验，不和仓库原始 README 混在一起。

## 整体目标

第一版实现走的是“第三种思路”的轻量版本：

- 整体流程固定
- 局部决策可学习
- 不做端到端强化学习
- 先离线收集候选模型输出
- 再训练一个轻量决策层

它的核心不是训练新的时间序列 backbone，而是在多个基础预测器之上学习：

1. 当前样本更适合哪些模型
2. 当前预测结果是否高风险
3. 高风险时是否需要做修正融合

## 新增文件

- `exp/exp_agentic_flow_forecasting.py`
  负责整个实验流程，包含离线缓存、meta-policy 训练和测试。
- `models/AgenticFlow.py`
  轻量决策层，包含路由、风险判断和修正融合。
- `utils/state_features.py`
  从输入窗口抽取状态特征。
- `utils/candidate_pool.py`
  读取 manifest，加载多个预训练基础模型，并统一做推理。
- `utils/meta_cache.py`
  将离线候选预测结果缓存到磁盘，避免重复跑大模型。
- `utils/case_bank.py`
  检索历史相似样本的最优模型与误差统计，作为修正先验。
- `configs/agentic_flow/sample_manifest.json`
  候选模型配置样例。
- `scripts/agentic_flow_forecast/sample_full.sh`
  一条完整运行命令样例。

## 整体逻辑

整个流程分为三步。

### 第一步：离线收集 meta cache

入口在 `Exp_Agentic_Flow_Forecast._collect_meta_cache()`。

对 `train / val / test` 三个 split 分别做一遍：

1. 读取原始时间序列 batch
2. 计算状态特征
3. 用 5 个基础模型分别预测
4. 计算每个模型在该样本上的误差
5. 记录样本级最优模型
6. 计算模型分歧
7. 构造“是否值得修正”的风险标签
8. 把这些结果保存到 `meta_cache/*.pt`

这样做的目的，是把“昂贵的大模型前向”变成一次性离线处理。后续训练路由器和融合头时，直接读缓存，不再重复跑 5 个大模型。

### 第二步：训练轻量决策层

入口在 `Exp_Agentic_Flow_Forecast._train_meta_policy()`。

训练阶段使用第一步生成的缓存，不直接读原始序列。

模型 `AgenticFlow` 做三件事：

1. `router`
   输入状态特征，预测当前样本更适合哪些基础模型。
2. `risk verifier`
   输入状态特征、候选模型分歧和历史案例先验，判断当前样本是否高风险。
3. `revision fusion`
   如果风险较高，则重新分配候选模型权重，得到修正后的预测。

训练目标由三部分组成：

- 路由损失：让 router 尽量接近样本级最优模型
- 风险损失：让 verifier 学会哪些样本值得修正
- 融合损失：让最终融合结果的 MSE 尽可能低

### 第三步：测试推理

入口在 `Exp_Agentic_Flow_Forecast.test()`。

测试时不再跑基础模型训练，只读取：

- 已保存的 meta-policy checkpoint
- 已保存的 test cache
- train cache 构造的案例库

推理逻辑是：

1. 读取当前样本的状态特征和候选模型预测
2. 路由器输出 top-k 候选模型
3. 案例库返回相似样本的模型先验和误差先验
4. 风险模块输出风险分数
5. 如果风险分数高于阈值，就使用修正融合结果
6. 否则直接使用初始路由结果

最后输出：

- `pred.npy`
- `true.npy`
- `weights.npy`
- `risk_scores.npy`
- `metrics.npy`

## 状态特征是什么

`utils/state_features.py` 第一版实现的是手工统计特征，不是新的深层编码器。这样做更稳，也更方便解释。

当前特征包括：

- 全局均值
- 全局标准差
- 绝对均值
- 最后一步变化幅度
- 起点和终点的均值差
- 趋势斜率
- 一阶差分波动率
- 一阶自相关
- 季节性滞后自相关
- 主频能量占比
- 通道相关性均值

这些特征的作用是回答一个问题：

“当前样本属于什么时序状态？”

## 风险标签怎么构造

第一版没有用复杂规则，而是采用简单、稳定的离线定义。

对每个样本：

1. 计算每个候选模型的样本级误差
2. 找到最佳单模型误差
3. 再构造一个 oracle 融合结果
4. 如果 oracle 融合显著优于最佳单模型，就把这个样本标为“值得修正”

这样做的意义是：

- 让 verifier 学会识别“单模型不够稳”的样本
- 不需要额外的人为标注

## 案例库是什么逻辑

`utils/case_bank.py` 是一个非常轻量的检索式 memory。

它不保存原始时间序列全文，只保存：

- 状态特征
- 样本级最优模型
- 各个候选模型的误差

查询时做的事情也很简单：

1. 用当前状态特征去找最相似的历史样本
2. 统计这些相似样本最常胜出的模型
3. 统计这些样本上各模型的平均误差

这些统计量作为先验送入风险模块和融合模块。

## manifest 怎么写

示例在 `configs/agentic_flow/sample_manifest.json`。

每个候选模型至少要提供：

- `name`
- `checkpoint`

可选提供：

- `overrides`
  只改 `args` 里的普通字段
- `constructor_kwargs`
  传给模型构造函数的额外参数

最关键的一点是：

`checkpoint` 必须指向你已经训练好的基础模型权重。

因为这一版不是端到端 joint training，而是建立在已有基础模型池上的二阶段实验。

## 如何运行

建议按下面顺序。

### 1. 先训练基础模型

至少先准备这 5 个模型的 checkpoint：

- DLinear
- PatchTST
- TimesNet
- iTransformer
- TimeMixer

### 2. 修改 manifest

把 `configs/agentic_flow/sample_manifest.json` 里的占位 checkpoint 路径改成真实路径。

### 3. 收集离线缓存

如果你只想先做离线缓存：

```bash
python -u run.py --task_name agentic_flow_forecast --is_training 1 --model AgenticFlow --agentic_stage collect_meta ...
```

### 4. 训练 meta-policy

```bash
python -u run.py --task_name agentic_flow_forecast --is_training 1 --model AgenticFlow --agentic_stage train_meta ...
```

### 5. 一次性全流程运行

也可以直接用：

```bash
python -u run.py --task_name agentic_flow_forecast --is_training 1 --model AgenticFlow --agentic_stage full ...
```

对应样例脚本在：

- `scripts/agentic_flow_forecast/sample_full.sh`

## 关键参数说明

- `candidate_manifest`
  候选模型池配置文件。
- `meta_cache_dir`
  离线缓存目录。
- `top_k_candidates`
  路由后保留多少个候选模型。
- `risk_threshold`
  风险分数超过这个阈值时才触发修正。
- `case_topn`
  案例库检索时取多少个最近邻。
- `meta_hidden`
  轻量决策层隐藏维度。
- `meta_lr`
  决策层学习率。
- `route_loss_weight`
  路由损失权重。
- `risk_loss_weight`
  风险损失权重。
- `revision_margin`
  只有 oracle 融合比最佳单模型好到一定幅度，才把样本标为“值得修正”。

## 第一版实现的边界

当前版本是“先跑通、先做实验”的版本，不是最终论文版。

目前的设计边界有：

- 不做端到端强化学习
- 不做在线案例更新
- 不做复杂记忆网络
- 风险标签是离线构造的启发式标签
- 状态特征是手工统计特征

但这版已经足够支持下面几类实验：

- 单模型 vs 状态驱动决策
- 静态集成 vs 状态驱动修正
- 去掉案例库、去掉风险模块、去掉路由器的消融
- top-k 候选池大小消融

## 推荐实验顺序

如果只是先验证代码和逻辑，建议按这个顺序：

1. 先用 `ETTh1` 跑通
2. 再上 `Electricity`
3. 再扩到 `Traffic` 和 `Weather`

第一版的目标不是追求最好结果，而是先确认：

- cache 流程正确
- 5 个候选模型能被正确加载
- 路由、风险、修正逻辑是连通的
- 最终结果能稳定输出
