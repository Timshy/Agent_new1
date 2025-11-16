#!/bin/bash
# Agent.py 运行示例脚本
# 固定每轮30张抽样 + 极严苛评分（0.0-10.0分制）

# 激活虚拟环境（如果使用）
# source .venv/bin/activate

# 设置工作目录
cd "$(dirname "$0")"

# 运行 agent.py
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
  --api-key "${KIMI_API_KEY}" \
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

echo ""
echo "======================================"
echo "Agent 运行完成！"
echo "======================================"
echo "最佳模型: ./work/best_model.pth"
echo "轮次摘要: ./work/rounds_summary.json"
echo "备份目录: ./backup"
echo "可疑样本: ./suspicious_score_10"
echo ""
echo "启动监控面板："
echo "  streamlit run dashboard"
