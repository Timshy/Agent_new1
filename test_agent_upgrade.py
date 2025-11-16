#!/usr/bin/env python3
"""
Agent.py 升级验证脚本
验证核心功能：固定30张抽样 + 极严苛评分prompt
"""

import sys
import re
from pathlib import Path


def test_fixed_sampling():
    """测试1: 固定30张抽样逻辑"""
    print("=" * 60)
    print("测试1: 固定30张抽样逻辑")
    print("=" * 60)
    
    agent_path = Path("agent.py")
    if not agent_path.exists():
        print("❌ agent.py 不存在")
        return False
    
    content = agent_path.read_text(encoding='utf-8')
    
    # 检查1: 是否移除了 sample_ratio
    if "sample_ratio" in content.lower() and "--sample-ratio" in content:
        print("⚠️  警告: 代码中仍存在 sample_ratio 参数")
    
    # 检查2: 是否添加了 samples_per_round
    if "--samples-per-round" not in content:
        print("❌ 未找到 --samples-per-round 参数")
        return False
    print("✅ 找到 --samples-per-round 参数")
    
    # 检查3: 固定抽样逻辑
    if "min(args.samples_per_round, len(available_images))" not in content:
        print("❌ 未找到固定抽样逻辑：min(args.samples_per_round, len(available_images))")
        return False
    print("✅ 找到固定抽样逻辑")
    
    # 检查4: 随机种子设置
    if "random.seed(round_num)" not in content:
        print("⚠️  警告: 未找到 random.seed(round_num)，可能影响可复现性")
    else:
        print("✅ 找到随机种子设置（可复现）")
    
    print("\n✅ 测试1通过：固定30张抽样逻辑已正确实现\n")
    return True


def test_strict_prompt():
    """测试2: 极严苛评分Prompt"""
    print("=" * 60)
    print("测试2: 极严苛评分Prompt")
    print("=" * 60)
    
    agent_path = Path("agent.py")
    content = agent_path.read_text(encoding='utf-8')
    
    # 检查1: 0.0-10.0 分制
    if "0.0–10.0" not in content and "0.0-10.0" not in content:
        print("❌ 未找到 0.0-10.0 评分范围")
        return False
    print("✅ 找到 0.0-10.0 评分范围")
    
    # 检查2: 禁止9.0及以上
    if "禁止给 9.0 及以上" not in content and "禁止 9.0" not in content:
        print("❌ 未找到禁止9.0及以上的约束")
        return False
    print("✅ 找到禁止9.0及以上约束")
    
    # 检查3: 扣分细则
    keywords = ["边界误差", "欠分割", "过分割", "空洞/噪点"]
    missing = [kw for kw in keywords if kw not in content]
    if missing:
        print(f"❌ 扣分细则缺失: {missing}")
        return False
    print("✅ 找到完整扣分细则（边界/欠分割/过分割/空洞）")
    
    # 检查4: temperature=0.0
    if '"temperature": 0.0' not in content:
        print("⚠️  警告: temperature 不是 0.0，可能影响确定性")
    else:
        print("✅ temperature=0.0（确保确定性）")
    
    # 检查5: 最高分 8.9
    if "8.9" not in content or "8.7–8.9" not in content:
        print("⚠️  警告: 未明确说明最高分 8.9")
    else:
        print("✅ 最高分 8.9 已说明")
    
    print("\n✅ 测试2通过：极严苛评分Prompt已正确嵌入\n")
    return True


def test_suspicious_audit():
    """测试3: 高分样本审计"""
    print("=" * 60)
    print("测试3: 高分样本审计（>=9.0）")
    print("=" * 60)
    
    agent_path = Path("agent.py")
    content = agent_path.read_text(encoding='utf-8')
    
    # 检查1: suspicious_root 参数
    if "--suspicious-root" not in content:
        print("❌ 未找到 --suspicious-root 参数")
        return False
    print("✅ 找到 --suspicious-root 参数")
    
    # 检查2: 审计逻辑（分数 >= 9.0）
    if "result[\"overall_score\"] >= 9.0" not in content:
        print("❌ 未找到高分审计逻辑（>= 9.0）")
        return False
    print("✅ 找到高分审计逻辑（>= 9.0）")
    
    # 检查3: 可疑样本保存
    if "suspicious_name" not in content or "suspicious_dir" not in content:
        print("⚠️  警告: 可疑样本保存逻辑可能不完整")
    else:
        print("✅ 可疑样本保存逻辑完整")
    
    print("\n✅ 测试3通过：高分审计功能已实现\n")
    return True


def test_warm_start():
    """测试4: 热启动训练"""
    print("=" * 60)
    print("测试4: 热启动训练")
    print("=" * 60)
    
    agent_path = Path("agent.py")
    content = agent_path.read_text(encoding='utf-8')
    
    # 检查1: pretrained_model 参数
    if "pretrained_model" not in content:
        print("❌ 未找到 pretrained_model 参数")
        return False
    print("✅ 找到 pretrained_model 参数")
    
    # 检查2: 学习率递减
    if "0.8 ** round_num" not in content:
        print("❌ 未找到学习率递减逻辑（0.8 ** round_num）")
        return False
    print("✅ 找到学习率递减逻辑（0.8 ** round_num）")
    
    # 检查3: 热启动逻辑
    if "best_model.pth" not in content:
        print("⚠️  警告: 未找到 best_model.pth 引用")
    else:
        print("✅ 找到 best_model.pth 热启动逻辑")
    
    print("\n✅ 测试4通过：热启动训练已实现\n")
    return True


def test_enhanced_logging():
    """测试5: 增强日志字段"""
    print("=" * 60)
    print("测试5: 增强日志字段")
    print("=" * 60)
    
    agent_path = Path("agent.py")
    content = agent_path.read_text(encoding='utf-8')
    
    # 检查新增字段
    required_fields = [
        "sampled_count",
        "avg_score_this_round",
        "high_confidence_rate"
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in content:
            missing_fields.append(field)
        else:
            print(f"✅ 找到日志字段: {field}")
    
    if missing_fields:
        print(f"❌ 缺失日志字段: {missing_fields}")
        return False
    
    print("\n✅ 测试5通过：增强日志字段已添加\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Agent.py 升级验证测试")
    print("=" * 60 + "\n")
    
    tests = [
        test_fixed_sampling,
        test_strict_prompt,
        test_suspicious_audit,
        test_warm_start,
        test_enhanced_logging
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append(False)
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有测试通过！Agent.py 升级成功！")
        return 0
    else:
        print("\n⚠️  部分测试未通过，请检查代码")
        return 1


if __name__ == "__main__":
    sys.exit(main())
