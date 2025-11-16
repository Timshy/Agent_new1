# Agent.py 升级说明 - 固定30张抽样 + 极严苛评分

## 核心升级内容

### ✅ 1. 固定每轮预测样本数（N=30）

**旧版本**：按百分比抽样（`sample_ratio=0.2`，即每轮抽取剩余图像的20%）

**新版本**：固定数量抽样
- **每轮抽取 `min(30, 剩余图像数)` 张**
- 新增参数：`--samples-per-round 30`（默认30）
- 抽样逻辑：
  ```python
  random.seed(round_num)  # 保证可复现
  sample_size = min(args.samples_per_round, len(available_images))
  sampled_images = random.sample(available_images, sample_size)
  ```
- **无论是否入选伪标签集，都永久删除这30张**（移至 `backup/round_X/`）
- 若剩余图像为0 → 自动终止循环

### ✅ 2. 极严苛 Kimi 评分 Prompt

**旧版本**：0-100 分制，相对宽松（90-100=优秀）

**新版本**：0.0-10.0 分制，极严苛（最高8.9，禁止9.0及以上）

**新 Prompt 核心特性**：
```text
你是顶尖医学图像分割质检AI，任务是对预测掩码进行 **0.0–10.0 评分（保留1位小数，禁止给 9.0 及以上）**。

【扣分细则】（必须逐项检查，零容忍）
1. **边界误差**（权重50%）：任意位置边界偏差 >1px 扣1.0分，>3px 扣2.5分，存在锯齿/毛刺 扣1.5分
2. **欠分割**（权重25%）：遗漏真实结构 >5% 扣2.0分，>10% 扣4.0分
3. **过分割**（权重15%）：假阳性区域 >3% 扣1.5分，>8% 扣3.0分
4. **空洞/噪点**（权重10%）：前景内空洞 >3px 扣1.0分，孤立噪点 >5个 扣0.8分

【输出要求】
- 必须返回 **纯 JSON**：{"overall_score": 7.6, "reason": "边界轻微锯齿，欠分割约6%"}
- 最高分 8.9，完美分割给 8.7–8.9
- reason 控制在 12字以内
- temperature=0.0，确保确定性
```

**评分范围调整**：
- 旧版：`--high-confidence 80.0`（80/100）
- 新版：`--high-confidence 7.5`（7.5/10）

### ✅ 3. 高分样本审计机制

新增 `--suspicious-root` 参数，自动保存 **分数 >=9.0 的可疑样本**：
```bash
suspicious_score_10/
├── images/          # 原始图像
└── pred_masks/      # 预测掩码
```

虽然 Prompt 要求禁止给9.0及以上，但若 API 返回异常高分，系统会自动审计。

### ✅ 4. 热启动训练（Warm Start）

每轮训练**从上一轮最佳模型继续微调**：
```python
pretrained = work_root / "best_model.pth" if exists else init_model_path
train_unet(..., pretrained_model=pretrained)
```

**学习率递减策略**：
```python
lr = args.lr * (0.8 ** round_num)
```
- Round 1: `1e-5 * 0.8^1 = 8e-6`
- Round 2: `1e-5 * 0.8^2 = 6.4e-6`
- Round 5: `1e-5 * 0.8^5 = 3.28e-6`

### ✅ 5. 增强日志字段

新增监控指标：
```json
{
  "sampled_count": 30,              // 本轮抽样数量
  "avg_score_this_round": 7.6,      // 本轮平均评分
  "high_confidence_rate": 0.4667,   // 高置信率（入选/抽样）
  "selected": 14,
  "pseudo_total": 120,
  "dice": 0.8532
}
```

---

## 运行示例

### 基础命令（推荐）

```bash
python agent.py \
  --train-dataset-dir ./data/train_pool/images \
  --labeled-img-dir ./data/labeled/images \
  --labeled-mask-dir ./data/labeled/masks \
  --test-img-dir ./data/test/images \
  --test-mask-dir ./data/test/masks \
  --api-key "sk-xxx" \
  --high-confidence 7.5 \
  --max-rounds 50 \
  --samples-per-round 30 \
  --max-pseudo-samples 5000 \
  --backup-root ./backup \
  --suspicious-root ./suspicious_score_10
```

### 完整参数（高级）

```bash
python agent.py \
  --train-dataset-dir ./data/train_pool/images \
  --labeled-img-dir ./data/labeled/images \
  --labeled-mask-dir ./data/labeled/masks \
  --test-img-dir ./data/test/images \
  --test-mask-dir ./data/test/masks \
  --work-root ./work \
  --backup-root ./backup \
  --suspicious-root ./suspicious_score_10 \
  --unet-script ./unet \
  --api-key "sk-xxx" \
  --samples-per-round 30 \
  --high-confidence 7.5 \
  --max-rounds 50 \
  --max-pseudo-samples 5000 \
  --seed 42 \
  --lr 1e-5 \
  --batch-size 4 \
  --epoch 10 \
  --dice-weight 0.6 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --use-amp
```

---

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--samples-per-round` | 30 | **每轮固定抽样数量**（核心改动） |
| `--high-confidence` | 7.5 | **高置信阈值（0-10）**（适配新评分） |
| `--max-rounds` | 50 | 最大迭代轮次 |
| `--max-pseudo-samples` | 5000 | 伪标签集最大容量（FIFO） |
| `--suspicious-root` | None | 可疑高分样本保存目录（>=9.0） |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 1e-5 | 初始学习率（自动递减：`lr * 0.8^round`） |
| `--batch-size` | 4 | 批大小 |
| `--epoch` | 10 | 每轮训练轮数 |
| `--use-amp` | False | 启用混合精度训练（加速） |

---

## 输出文件结构

```
work/
├── best_model.pth                  # 全局最佳模型（热启动源）
├── rounds_summary.json             # 轮次摘要
├── final_summary.json              # 最终总结
├── monitor/                        # Streamlit 监控数据
│   ├── metrics_history.json
│   ├── current_status.json
│   └── chart_data.json
├── pseudo_labels/                  # 伪标签集（FIFO 控制）
│   ├── images/
│   ├── masks/
│   └── pseudo_metadata.json
└── round_X_model/                  # 每轮模型

backup/
└── round_X/                        # 每轮备份（永久删除的图像）
    ├── images/
    ├── pred_masks/
    ├── scores.json
    └── selected.txt

suspicious_score_10/                # 可疑高分样本（>=9.0）
├── images/
└── pred_masks/
```

---

## 关键特性

### 1. 滚动抽样 + 永久删除
- 每轮抽取 30 张 → 预测 → 评分
- **无论是否入选，都永久删除**（避免重复）
- 剩余为0时自动终止

### 2. 极严苛评分
- 0.0-10.0 分制，最高 8.9
- 边界误差 >1px 扣分，零容忍
- Temperature=0.0，确保确定性

### 3. 热启动微调
- 每轮从 `best_model.pth` 继续训练
- 学习率指数递减：`lr * 0.8^round`
- 快速收敛，避免灾难性遗忘

### 4. 可审计终止
- 高分（>=9.0）样本自动保存至 `suspicious_score_10/`
- 备份目录保留所有抽样历史
- 完整日志：`sampled_count`, `avg_score_this_round`, `high_confidence_rate`

---

## 与旧版本对比

| 维度 | 旧版本 | 新版本 |
|------|--------|--------|
| 抽样策略 | 按比例（20%） | **固定数量（30张）** |
| 评分范围 | 0-100 | **0.0-10.0（极严苛）** |
| 评分 Prompt | 相对宽松 | **零容忍扣分细则** |
| 热启动 | 不支持 | **从 best_model.pth 继续** |
| 学习率 | 固定 | **指数递减（0.8^round）** |
| 高分审计 | 无 | **>=9.0 自动保存** |
| 日志字段 | 基础 | **新增 sampled_count, high_confidence_rate** |

---

## 最佳实践

### 1. 高置信阈值设置
- **保守**：`--high-confidence 8.0`（仅接受近完美分割）
- **平衡**：`--high-confidence 7.5`（推荐）
- **激进**：`--high-confidence 7.0`（快速扩展伪标签）

### 2. 学习率策略
- **初始模型**：`--lr 1e-4`（从头训练）
- **热启动微调**：`--lr 1e-5`（推荐，自动递减）
- **精细调优**：`--lr 5e-6`

### 3. 终止条件
- **数据耗尽**：Train_Dataset 为空
- **性能停滞**：`best_dice` 连续 5 轮未提升（手动监控）
- **目标达成**：`best_dice >= 0.90`

---

## FAQ

**Q1: 为什么固定30张而不是百分比？**
- 确保每轮评估成本可控（30次 API 调用）
- 避免后期剩余样本过少导致抽样不稳定

**Q2: 如何处理剩余图像 <30 的情况？**
- 自动抽取全部剩余图像：`min(30, len(available))`
- 完成后若目录为空，终止循环

**Q3: 为什么禁止给9.0及以上？**
- 极严苛评分防止"分数通胀"
- 理论最高8.9，确保入选样本确实高质量
- >=9.0 样本自动进入审计流程

**Q4: 热启动会导致过拟合吗？**
- 学习率递减策略缓解（0.8^round）
- 伪标签集动态更新（FIFO 5000张）
- 测试集持续监控 Dice，避免性能下降

**Q5: UNet 脚本需要支持 `--pretrained_model` 参数吗？**
- 是的！请确保 `unet` 脚本支持热启动：
  ```python
  if args.pretrained_model and Path(args.pretrained_model).exists():
      model.load_state_dict(torch.load(args.pretrained_model))
  ```

---

## 日本时间（JST）研发级特性

✅ **固定30张抽样**：精准控制每轮成本  
✅ **极严苛评分**：0.0-10.0 分制，零容忍扣分  
✅ **热启动微调**：从 best_model.pth 继续，学习率递减  
✅ **高分审计**：>=9.0 自动保存，可复盘异常  
✅ **可复现**：`random.seed(round)` 确保抽样一致  
✅ **增强日志**：`sampled_count`, `avg_score_this_round`, `high_confidence_rate`  
✅ **完整备份**：所有抽样数据永久保存至 `backup/`  

**准备好自演化了吗？🚀**
