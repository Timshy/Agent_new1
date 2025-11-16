# Agent.py 快速启动指南

## 🚀 一键启动（推荐）

### 使用示例脚本
```bash
# 1. 设置API密钥
export KIMI_API_KEY="sk-xxx"

# 2. 运行（使用默认路径）
./run_agent_example.sh
```

### 手动运行（自定义路径）
```bash
python agent.py \
  --train-dataset-dir ./data/train_pool/images \
  --labeled-img-dir ./data/labeled/images \
  --labeled-mask-dir ./data/labeled/masks \
  --test-img-dir ./data/test/images \
  --test-mask-dir ./data/test/masks \
  --api-key "sk-xxx" \
  --samples-per-round 30 \
  --high-confidence 7.5 \
  --max-rounds 50
```

---

## 📊 核心参数速查

### 必需参数
```bash
--train-dataset-dir PATH      # 待标注图像池（每轮抽取30张）
--labeled-img-dir PATH        # 初始标注图像
--labeled-mask-dir PATH       # 初始标注掩码
--test-img-dir PATH           # 测试集图像
--test-mask-dir PATH          # 测试集掩码
--api-key "sk-xxx"            # Kimi API密钥
```

### 关键参数（推荐调整）
```bash
--samples-per-round 30        # 每轮固定抽样数（默认30）
--high-confidence 7.5         # 高置信阈值（0-10，默认7.5）
--max-rounds 50               # 最大迭代轮次（默认50）
--max-pseudo-samples 5000     # 伪标签集容量（FIFO，默认5000）
```

### 可选参数
```bash
--work-root ./work            # 工作目录（模型、日志等）
--backup-root ./backup        # 备份目录（每轮抽样数据）
--suspicious-root ./suspicious_score_10  # 可疑高分样本（>=9.0）
--lr 1e-5                     # 初始学习率（自动递减）
--use-amp                     # 启用混合精度训练
```

---

## 🎯 新版本核心特性

### ✅ 固定每轮30张抽样
- **旧版**：按比例抽样（不稳定，成本不可控）
- **新版**：固定30张（可控、可复现）
- 剩余不足30张时抽取全部，完成后自动终止

### ✅ 极严苛评分（0.0-10.0）
- **最高分**：8.9（禁止9.0及以上）
- **扣分细则**：边界误差（50%）、欠分割（25%）、过分割（15%）、空洞/噪点（10%）
- **确定性**：temperature=0.0

### ✅ 热启动训练
- 每轮从 `work/best_model.pth` 继续训练
- 学习率递减：`lr * (0.8 ** round)`
- 快速收敛，避免灾难性遗忘

### ✅ 高分审计
- 分数 >=9.0 的样本自动保存至 `suspicious_score_10/`
- 包含原图和预测掩码，便于人工复核

---

## 📁 输出文件结构

```
work/
├── best_model.pth              ⭐ 全局最佳模型
├── rounds_summary.json         📊 轮次摘要
├── final_summary.json          📋 最终总结
├── monitor/                    📈 Streamlit监控数据
│   ├── metrics_history.json
│   └── current_status.json
├── pseudo_labels/              🏷️ 伪标签集（FIFO 5000张）
│   ├── images/
│   ├── masks/
│   └── pseudo_metadata.json
└── round_X_model/              🔄 每轮模型

backup/
└── round_X/                    💾 每轮备份
    ├── images/                 （永久删除的图像）
    ├── pred_masks/
    ├── scores.json
    └── selected.txt

suspicious_score_10/            🔍 可疑高分样本
├── images/
└── pred_masks/
```

---

## 🔧 高置信阈值设置指南

| 策略 | 阈值 | 特点 | 适用场景 |
|-----|------|------|---------|
| **保守** | 8.0+ | 仅接受近完美分割 | 初期标注数据充足 |
| **平衡** | 7.5 | **推荐** | 通用场景 |
| **激进** | 7.0 | 快速扩展伪标签集 | 标注数据稀缺 |

**评分对照表**（旧版 0-100 → 新版 0-10）：
```
旧版 90+ → 新版 8.7-8.9  （完美）
旧版 80+ → 新版 7.5-8.6  （优秀，推荐阈值）
旧版 70+ → 新版 6.5-7.4  （良好）
旧版 60+ → 新版 5.5-6.4  （中等）
```

---

## 🧪 验证升级

运行测试脚本确认升级成功：
```bash
python test_agent_upgrade.py
```

预期输出：
```
============================================================
测试汇总
============================================================
通过: 5/5
🎉 所有测试通过！Agent.py 升级成功！
```

---

## 📖 详细文档

- **📘 完整使用说明**：[AGENT_UPGRADE_README.md](AGENT_UPGRADE_README.md)
- **📝 变更日志**：[UPGRADE_CHANGELOG.md](UPGRADE_CHANGELOG.md)
- **🧪 测试脚本**：`test_agent_upgrade.py`
- **🚀 运行示例**：`run_agent_example.sh`

---

## ⚡ 典型运行流程

```
初始化
└─ 使用初始标注集训练基础模型 (Round 0)
    └─ 保存至 work/round_0_initial/best_model.pth

Round 1
├─ 从 Train_Dataset 随机抽取 30 张 (seed=1)
├─ UNet 预测 → 生成掩码
├─ Kimi 评分（0.0-10.0，最高8.9）
│   └─ 平均分 7.6，14/30 入选（high_confidence_rate=0.467）
├─ 入选样本加入伪标签集（FIFO 控制容量）
├─ 备份至 backup/round_1/
├─ 永久删除这30张
├─ 热启动训练（从 best_model.pth，lr=1e-5*0.8^1=8e-6）
│   └─ 训练集 = 初始标注(100) + 伪标签(14) = 114 张
└─ 若 Dice 提升 → 更新 work/best_model.pth

Round 2
├─ 剩余 970 张，再抽 30 张 (seed=2)
├─ 热启动（从 Round 1 最佳模型，lr=6.4e-6）
└─ ...

Round N
└─ Train_Dataset 为空 → 自动终止
    └─ 输出最终总结至 work/final_summary.json
```

---

## 🔥 常见场景

### 场景1：快速实验（小数据集）
```bash
python agent.py \
  --train-dataset-dir ./data/train_pool/images \
  --labeled-img-dir ./data/labeled/images \
  --labeled-mask-dir ./data/labeled/masks \
  --test-img-dir ./data/test/images \
  --test-mask-dir ./data/test/masks \
  --api-key "sk-xxx" \
  --samples-per-round 10 \    # 减少抽样数
  --high-confidence 7.0 \      # 降低阈值
  --max-rounds 20 \            # 减少轮次
  --epoch 5                    # 减少训练轮数
```

### 场景2：生产级训练（大数据集）
```bash
python agent.py \
  --train-dataset-dir ./data/train_pool/images \
  --labeled-img-dir ./data/labeled/images \
  --labeled-mask-dir ./data/labeled/masks \
  --test-img-dir ./data/test/images \
  --test-mask-dir ./data/test/masks \
  --api-key "sk-xxx" \
  --samples-per-round 30 \
  --high-confidence 7.5 \
  --max-rounds 100 \
  --max-pseudo-samples 10000 \
  --epoch 15 \
  --use-amp                    # 混合精度加速
```

### 场景3：极严苛质控（高精度需求）
```bash
python agent.py \
  --train-dataset-dir ./data/train_pool/images \
  --labeled-img-dir ./data/labeled/images \
  --labeled-mask-dir ./data/labeled/masks \
  --test-img-dir ./data/test/images \
  --test-mask-dir ./data/test/masks \
  --api-key "sk-xxx" \
  --samples-per-round 30 \
  --high-confidence 8.0 \      # 极高阈值
  --max-rounds 50 \
  --suspicious-root ./suspicious_score_10  # 启用审计
```

---

## 🎓 最佳实践

1. **首次运行**：使用 `--max-rounds 5` 测试整个流程
2. **监控Dice**：若连续5轮未提升，考虑降低阈值或停止训练
3. **审查可疑样本**：定期检查 `suspicious_score_10/` 目录
4. **备份保留**：`backup/` 目录包含所有抽样历史，便于复盘
5. **热启动优势**：确保 `unet` 脚本支持 `--pretrained_model` 参数

---

## ❓ 常见问题

**Q: 如何查看实时进度？**
```bash
# 启动监控面板（另一个终端）
streamlit run dashboard
```

**Q: 如何恢复中断的训练？**
- Agent 会自动加载 `work/best_model.pth`
- 伪标签集通过 `pseudo_metadata.json` 恢复
- 从中断的轮次继续（需手动调整 `--max-rounds`）

**Q: 为什么有些样本评分很低？**
- 极严苛评分：边界偏差 >1px 即扣分
- 这是设计特性，确保入选样本高质量

**Q: 如何加速训练？**
1. 启用混合精度：`--use-amp`
2. 增大批大小：`--batch-size 8`（需GPU内存足够）
3. 减少训练轮数：`--epoch 5`

---

## 🚀 准备好开始了吗？

```bash
# 1. 设置API密钥
export KIMI_API_KEY="sk-xxx"

# 2. 运行
./run_agent_example.sh

# 3. 监控（另一个终端）
streamlit run dashboard
```

**祝您训练愉快！🎉**
