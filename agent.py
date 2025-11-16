import argparse
import base64
import json
import os
import random
import shutil
import subprocess
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


# ==================== 配置常量 ====================
API_URL = "https://api.moonshot.cn/v1/chat/completions"
MODEL = "kimi-latest"
API_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 5

ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = "imagenet"

SYSTEM_PROMPT = """你是顶尖医学图像分割质检AI，任务是对预测掩码进行 **0.0–10.0 评分（保留1位小数，禁止给 9.0 及以上）**。

【输入图像】
- 左侧：原始医学图像
- 右侧：模型预测的二值掩码（白色=前景）

【扣分细则】（必须逐项检查，零容忍）
1. **边界误差**（权重50%）：任意位置边界偏差 >1px 扣1.0分，>3px 扣2.5分，存在锯齿/毛刺 扣1.5分
2. **欠分割**（权重25%）：遗漏真实结构 >5% 扣2.0分，>10% 扣4.0分
3. **过分割**（权重15%）：假阳性区域 >3% 扣1.5分，>8% 扣3.0分
4. **空洞/噪点**（权重10%）：前景内空洞 >3px 扣1.0分，孤立噪点 >5个 扣0.8分

【输出要求】
- 必须返回 **纯 JSON**，无任何多余字符：
  ```json
  {"overall_score": 7.6, "reason": "边界轻微锯齿，欠分割约6%"}
  ```
- 最高分 8.9，完美分割给 8.7–8.9
- reason 控制在 12字以内
- 温度 temperature=0.0，确保确定性

现在开始评分：
"""


# ==================== 训练监控器 ====================
class TrainingMonitor:
    """实时训练监控和指标记录"""
    
    def __init__(self, monitor_dir: Path):
        self.monitor_dir = monitor_dir
        ensure_dir(self.monitor_dir)
        
        self.metrics_history_path = self.monitor_dir / "metrics_history.json"
        self.current_status_path = self.monitor_dir / "current_status.json"
        self.chart_data_path = self.monitor_dir / "chart_data.json"
        
        self.metrics_history = []
        self.load_history()
        
    def load_history(self) -> None:
        """从文件加载历史数据"""
        if self.metrics_history_path.exists():
            try:
                with open(self.metrics_history_path, 'r', encoding='utf-8') as f:
                    self.metrics_history = json.load(f)
            except Exception as e:
                print(f"[Monitor] 加载历史数据失败: {e}")
                self.metrics_history = []
    
    def record_round_metrics(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """记录单轮指标"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "round": round_num,
            **metrics
        }
        self.metrics_history.append(record)
        self._save_metrics_history()
    
    def update_status(self, status: Dict[str, Any]) -> None:
        """更新当前状态"""
        status['timestamp'] = datetime.now().isoformat()
        with open(self.current_status_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
    
    def _save_metrics_history(self) -> None:
        """保存指标历史"""
        with open(self.metrics_history_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
    
    def get_chart_data(self) -> Dict[str, Any]:
        """获取图表数据"""
        if not self.metrics_history:
            return {}
        
        rounds = [m['round'] for m in self.metrics_history]
        dice_scores = [m.get('dice', 0) for m in self.metrics_history]
        precision_scores = [m.get('precision', 0) for m in self.metrics_history]
        recall_scores = [m.get('recall', 0) for m in self.metrics_history]
        f1_scores = [m.get('f1', 0) for m in self.metrics_history]
        pseudo_counts = [m.get('pseudo_total', 0) for m in self.metrics_history]
        selected_counts = [m.get('selected', 0) for m in self.metrics_history]
        avg_confidences = [m.get('avg_confidence', 0) for m in self.metrics_history]
        
        return {
            "rounds": rounds,
            "dice": dice_scores,
            "precision": precision_scores,
            "recall": recall_scores,
            "f1": f1_scores,
            "pseudo_total": pseudo_counts,
            "selected": selected_counts,
            "avg_confidence": avg_confidences
        }


# ==================== 工具函数 ====================
def ensure_dir(path: Path) -> None:
    """安全创建目录"""
    path.mkdir(parents=True, exist_ok=True)


def safe_rmtree(path: Path, retries: int = 3) -> None:
    """安全删除目录（带重试）"""
    if not path.exists():
        return
    for i in range(retries):
        try:
            shutil.rmtree(path)
            return
        except Exception as e:
            if i == retries - 1:
                print(f"[警告] 无法删除 {path}: {e}")
            time.sleep(0.5)


def get_all_images(img_dir: Path) -> List[str]:
    """获取目录下所有图像文件名"""
    if not img_dir.exists():
        return []
    return sorted([
        f.name for f in img_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
    ])


def image_to_base64(path: Path) -> str:
    """图像转 Base64"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ==================== 数据集类 ====================
class SimpleDataset(Dataset):
    """简单图像数据集（用于预测）"""
    def __init__(self, img_dir: Path, img_list: List[str]):
        self.img_dir = img_dir
        self.img_list = img_list

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_name = self.img_list[idx]
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB").resize((256, 256))
        image = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image_tensor, img_name


# ==================== 伪标签训练集管理器 ====================
class PseudoLabelManager:
    """
    固定大小伪标签训练集管理器（FIFO 策略）
    
    特性：
    - 固定最大容量（max_pseudo_samples）
    - 超出时删除最早加入的样本
    - 记录每个样本的元数据（加入时间、置信分等）
    """
    def __init__(self, pseudo_img_dir: Path, pseudo_mask_dir: Path, max_samples: int = 5000):
        self.img_dir = pseudo_img_dir
        self.mask_dir = pseudo_mask_dir
        self.max_samples = max_samples
        self.samples_queue = deque()  # (img_name, timestamp, score)
        self.metadata_path = pseudo_img_dir.parent / "pseudo_metadata.json"
        
        ensure_dir(self.img_dir)
        ensure_dir(self.mask_dir)
        
        # 恢复已有样本
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.samples_queue = deque([(m['name'], m['timestamp'], m['score']) for m in metadata])
        
        print(f"[PseudoLabelManager] 初始化完成 | 当前样本数: {len(self.samples_queue)} | 最大容量: {self.max_samples}")

    def add_samples(self, samples: List[Dict[str, Any]]) -> int:
        """
        添加样本到伪标签集
        samples: [{"img_path": Path, "mask_path": Path, "score": float, "name": str}, ...]
        返回：实际添加的数量
        """
        added = 0
        for sample in samples:
            src_img = sample['img_path']
            src_mask = sample['mask_path']
            score = sample['score']
            base_name = sample['name']
            
            # 生成唯一文件名
            timestamp = int(time.time() * 1000)
            unique_name = f"pseudo_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
            
            # 复制文件
            dst_img = self.img_dir / unique_name
            dst_mask = self.mask_dir / unique_name
            shutil.copy(src_img, dst_img)
            shutil.copy(src_mask, dst_mask)
            
            # 加入队列
            self.samples_queue.append((unique_name, timestamp, score))
            added += 1
            
            # FIFO：超出容量则删除最早的样本
            while len(self.samples_queue) > self.max_samples:
                old_name, _, _ = self.samples_queue.popleft()
                old_img = self.img_dir / old_name
                old_mask = self.mask_dir / old_name
                old_img.unlink(missing_ok=True)
                old_mask.unlink(missing_ok=True)
        
        # 保存元数据
        self._save_metadata()
        print(f"[PseudoLabelManager] 新增 {added} 个样本 | 当前总数: {len(self.samples_queue)}")
        return added

    def _save_metadata(self) -> None:
        """保存元数据到 JSON"""
        metadata = [{"name": n, "timestamp": t, "score": s} for n, t, s in self.samples_queue]
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def get_current_count(self) -> int:
        """获取当前样本数量"""
        return len(self.samples_queue)


# ==================== 预测 + 后处理 ====================
def predict_batch(
    model: torch.nn.Module,
    img_dir: Path,
    img_list: List[str],
    output_dir: Path,
    device: str
) -> Dict[str, Path]:
    """
    批量预测并保存预测掩码
    返回：{img_name: pred_mask_path}
    """
    model.eval()
    pred_dir = output_dir / "preds"
    ensure_dir(pred_dir)
    
    dataset = SimpleDataset(img_dir, img_list)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    predictions = {}
    
    with torch.no_grad():
        for images, img_names in tqdm(loader, desc="预测中", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            pred_np = preds[0, 0].cpu().numpy()
            
            # 简单二值化（阈值 0.5）
            binary = (pred_np > 0.5).astype(np.uint8) * 255
            
            img_name = img_names[0]
            pred_path = pred_dir / f"pred_{img_name}"
            Image.fromarray(binary).save(pred_path)
            predictions[img_name] = pred_path
    
    return predictions


# ==================== Kimi Vision API 调用 ====================
def call_kimi_vision(
    image_path: Path,
    pred_mask_path: Path,
    api_key: str,
    max_retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """
    调用 Kimi Vision API 评分（拼接图：原图 + 预测掩码）
    
    返回：{"overall_score": float, "reason": str, "success": bool}
    """
    # 拼接图像（左：原图 | 右：预测掩码）
    img = Image.open(image_path).convert("RGB").resize((256, 256))
    mask = Image.open(pred_mask_path).convert("RGB").resize((256, 256))
    
    concat_width = img.width + mask.width
    concat = Image.new("RGB", (concat_width, img.height))
    concat.paste(img, (0, 0))
    concat.paste(mask, (img.width, 0))
    
    # 转 Base64
    from io import BytesIO
    buffer = BytesIO()
    concat.save(buffer, format="JPEG")
    concat_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # 构造请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{concat_b64}"}
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=API_TIMEOUT)
            
            if response.status_code != 200:
                print(f"\n[Kimi API] 错误 {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return {"overall_score": 5.0, "reason": f"API错误: {response.status_code}", "success": False}
            
            data = response.json()
            reply = data["choices"][0]["message"]["content"].strip()
            
            # 提取 JSON
            start = reply.find("{")
            end = reply.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = reply[start:end]
                result = json.loads(json_str)
                
                # 验证必需字段
                if "overall_score" in result:
                    score = float(result["overall_score"])
                    reason = result.get("reason", "无说明")
                    return {"overall_score": score, "reason": reason, "success": True}
            
            print(f"\n[Kimi API] JSON 解析失败（尝试 {attempt+1}）: {reply}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            
        except Exception as e:
            print(f"\n[Kimi API] 请求异常（尝试 {attempt+1}）: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
    
    # 所有重试失败，返回默认分数
    return {"overall_score": 5.0, "reason": "API 调用失败", "success": False}


def batch_evaluate_with_kimi(
    image_dir: Path,
    predictions: Dict[str, Path],
    api_key: str,
    output_json: Path,
    suspicious_root: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    批量调用 Kimi API 评分
    
    suspicious_root: 可疑高分样本保存目录（>=9.0）
    返回：[{"image": str, "score": float, "reason": str}, ...]
    """
    if not api_key:
        print("[警告] 未配置 API_KEY，跳过 LLM 评分")
        return []
    
    results = []
    
    print(f"\n[Kimi 评分] 开始评估 {len(predictions)} 张图像...")
    
    # 创建可疑样本目录（如果需要）
    suspicious_dir = None
    if suspicious_root:
        suspicious_dir = suspicious_root
        ensure_dir(suspicious_dir / "images")
        ensure_dir(suspicious_dir / "pred_masks")
    
    for img_name, pred_path in tqdm(predictions.items(), desc="LLM 评分", leave=False):
        img_path = image_dir / img_name
        
        result = call_kimi_vision(img_path, pred_path, api_key)
        
        results.append({
            "image": img_name,
            "score": result["overall_score"],
            "reason": result["reason"],
            "success": result["success"]
        })
        
        # 检查是否需要保存可疑高分样本
        if suspicious_dir and result["overall_score"] >= 9.0:
            timestamp = int(time.time() * 1000)
            suspicious_name = f"suspicious_{timestamp}_{img_name}"
            
            shutil.copy(img_path, suspicious_dir / "images" / suspicious_name)
            shutil.copy(pred_path, suspicious_dir / "pred_masks" / suspicious_name)
            
            print(f"\n[审计] 发现可疑高分样本 ({result['overall_score']:.1f}): {img_name} → {suspicious_dir}")
        
        # 防止速率限制
        time.sleep(1)
    
    # 保存结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    avg_score = np.mean([r['score'] for r in results]) if results else 0.0
    success_rate = sum(r['success'] for r in results) / len(results) if results else 0.0
    print(f"[Kimi 评分] 完成 | 平均分: {avg_score:.2f} | 成功率: {success_rate:.1%}")
    
    return results


# ==================== 样本选择 ====================
def select_high_confidence_samples(
    scores: List[Dict[str, Any]],
    predictions: Dict[str, Path],
    image_dir: Path,
    threshold: float
) -> List[Dict[str, Any]]:
    """
    选择高置信度样本（score >= threshold）
    
    返回：[{"img_path": Path, "mask_path": Path, "score": float, "name": str}, ...]
    """
    selected = []
    
    for item in scores:
        if item['score'] >= threshold:
            img_name = item['image']
            pred_mask_path = predictions.get(img_name)
            img_path = image_dir / img_name
            
            if pred_mask_path and img_path.exists() and pred_mask_path.exists():
                selected.append({
                    "img_path": img_path,
                    "mask_path": pred_mask_path,
                    "score": item['score'],
                    "name": img_name
                })
    
    return selected


# ==================== 备份管理 ====================
def backup_round_data(
    round_num: int,
    sampled_images: List[str],
    image_dir: Path,
    predictions: Dict[str, Path],
    scores: List[Dict[str, Any]],
    selected_samples: List[Dict[str, Any]],
    backup_root: Path
) -> None:
    """
    备份当前轮次数据
    
    结构：
    backup/round_X/
    ├── images/          # 抽取的原始图像
    ├── pred_masks/      # 预测掩码
    ├── scores.json      # LLM 评分
    └── selected.txt     # 入选样本名单
    """
    round_backup_dir = backup_root / f"round_{round_num}"
    ensure_dir(round_backup_dir)
    
    img_backup_dir = round_backup_dir / "images"
    mask_backup_dir = round_backup_dir / "pred_masks"
    ensure_dir(img_backup_dir)
    ensure_dir(mask_backup_dir)
    
    # 备份图像和预测
    for img_name in sampled_images:
        src_img = image_dir / img_name
        dst_img = img_backup_dir / img_name
        if src_img.exists():
            shutil.copy(src_img, dst_img)
        
        if img_name in predictions:
            src_mask = predictions[img_name]
            dst_mask = mask_backup_dir / src_mask.name
            if src_mask.exists():
                shutil.copy(src_mask, dst_mask)
    
    # 备份评分
    scores_path = round_backup_dir / "scores.json"
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    
    # 备份入选名单
    selected_names = [s['name'] for s in selected_samples]
    selected_path = round_backup_dir / "selected.txt"
    with open(selected_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(selected_names))
    
    print(f"[备份] Round {round_num} 数据已备份 → {round_backup_dir}")


def remove_sampled_images(image_dir: Path, sampled_images: List[str]) -> None:
    """
    从 Train_Dataset 中永久删除已抽取的图像
    """
    removed_count = 0
    for img_name in sampled_images:
        img_path = image_dir / img_name
        if img_path.exists():
            img_path.unlink()
            removed_count += 1
    
    print(f"[清理] 已从 Train_Dataset 删除 {removed_count} 张图像")


# ==================== 训练封装 ====================
def train_unet(
    round_num: int,
    labeled_img_dir: Path,
    labeled_mask_dir: Path,
    pseudo_img_dir: Path,
    pseudo_mask_dir: Path,
    test_img_dir: Path,
    test_mask_dir: Path,
    save_dir: Path,
    unet_script: Path,
    args: argparse.Namespace,
    pretrained_model: Optional[Path] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    训练 UNet 模型（初始标注集 + 伪标签集）
    
    pretrained_model: 预训练模型路径（热启动）
    返回：(success: bool, metrics: dict)
    """
    # 合并训练集（临时目录）
    merged_img_dir = save_dir / "merged_train" / "images"
    merged_mask_dir = save_dir / "merged_train" / "masks"
    safe_rmtree(merged_img_dir.parent)
    ensure_dir(merged_img_dir)
    ensure_dir(merged_mask_dir)
    
    # 复制初始标注数据
    labeled_count = 0
    for img_path in labeled_img_dir.iterdir():
        if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            shutil.copy(img_path, merged_img_dir / img_path.name)
            
            # 查找对应 mask
            base = img_path.stem.replace("_img", "")
            for suffix in ["_mask.jpg", "mask.jpg", "_mask.png", "mask.png"]:
                mask_path = labeled_mask_dir / f"{base}{suffix}"
                if mask_path.exists():
                    shutil.copy(mask_path, merged_mask_dir / mask_path.name)
                    labeled_count += 1
                    break
    
    # 复制伪标签数据
    pseudo_count = 0
    if pseudo_img_dir.exists():
        for img_path in pseudo_img_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                shutil.copy(img_path, merged_img_dir / img_path.name)
                
                mask_path = pseudo_mask_dir / img_path.name
                if mask_path.exists():
                    shutil.copy(mask_path, merged_mask_dir / mask_path.name)
                    pseudo_count += 1
    
    print(f"[训练集] Round {round_num} | 初始标注: {labeled_count} | 伪标签: {pseudo_count} | 总计: {labeled_count + pseudo_count}")
    
    # 调用 unet 脚本
    # 学习率递减策略
    lr = args.lr * (0.8 ** round_num)
    
    cmd = [
        "python", str(unet_script),
        f"--lr={lr}",
        f"--batch_size={args.batch_size}",
        f"--epoch={args.epoch}",
        f"--encoder_name={ENCODER_NAME}",
        f"--encoder_weights={ENCODER_WEIGHTS}",
        f"--dice_weight={args.dice_weight}",
        f"--weight_decay={args.weight_decay}",
        f"--num_workers={args.num_workers}",
        f"--save_dir={save_dir}",
        f"--train_img_dir={merged_img_dir}",
        f"--train_mask_dir={merged_mask_dir}",
        f"--test_img_dir={test_img_dir}",
        f"--test_mask_dir={test_mask_dir}"
    ]
    
    # 热启动：从预训练模型继续
    if pretrained_model and pretrained_model.exists():
        cmd.append(f"--pretrained_model={pretrained_model}")
        print(f"[训练] Round {round_num} 热启动：{pretrained_model}")
    
    if args.use_amp:
        cmd.append("--use_amp")
    
    print(f"[训练] Round {round_num} 开始训练... (lr={lr:.2e})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[训练] Round {round_num} 训练失败！")
        print(result.stderr[-1000:] if result.stderr else "")
        return False, {}
    
    # 读取指标
    metrics_path = save_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_history = json.load(f)
            # 取最后一个 epoch 的指标
            if metrics_history:
                last_metrics = metrics_history[-1]
                return True, last_metrics
    
    return True, {}


# ==================== 主流程 ====================
def run_rolling_evolution(args: argparse.Namespace) -> None:
    """
    滚动抽样自演化主流程（固定每轮30张）
    """
    # ==================== 初始化路径 ====================
    train_dataset_dir = Path(args.train_dataset_dir).expanduser().resolve()
    labeled_img_dir = Path(args.labeled_img_dir).expanduser().resolve()
    labeled_mask_dir = Path(args.labeled_mask_dir).expanduser().resolve()
    test_img_dir = Path(args.test_img_dir).expanduser().resolve()
    test_mask_dir = Path(args.test_mask_dir).expanduser().resolve()
    backup_root = Path(args.backup_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    unet_script = Path(args.unet_script).expanduser().resolve()
    suspicious_root = Path(args.suspicious_root).expanduser().resolve() if args.suspicious_root else None
    
    # 伪标签目录
    pseudo_root = work_root / "pseudo_labels"
    pseudo_img_dir = pseudo_root / "images"
    pseudo_mask_dir = pseudo_root / "masks"
    
    # 确保目录存在
    ensure_dir(backup_root)
    ensure_dir(work_root)
    if suspicious_root:
        ensure_dir(suspicious_root)
    
    # 检查必要文件
    if not unet_script.exists():
        raise FileNotFoundError(f"UNet 脚本不存在: {unet_script}")
    
    if not labeled_img_dir.exists() or not labeled_mask_dir.exists():
        raise FileNotFoundError("初始标注集目录不存在！")
    
    if not train_dataset_dir.exists():
        raise FileNotFoundError(f"Train_Dataset 目录不存在: {train_dataset_dir}")
    
    # ==================== 初始化组件 ====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"滚动抽样自演化系统启动（固定每轮 {args.samples_per_round} 张）")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"Train_Dataset: {train_dataset_dir}")
    print(f"初始标注集: {labeled_img_dir}")
    print(f"测试集: {test_img_dir}")
    print(f"最大轮次: {args.max_rounds}")
    print(f"每轮抽样数: {args.samples_per_round}")
    print(f"高置信阈值: {args.high_confidence:.1f}/10.0")
    print(f"伪标签最大数量: {args.max_pseudo_samples}")
    if suspicious_root:
        print(f"可疑样本审计: {suspicious_root}")
    print(f"{'='*60}\n")
    
    # 初始化伪标签管理器
    pseudo_manager = PseudoLabelManager(pseudo_img_dir, pseudo_mask_dir, args.max_pseudo_samples)
    
    # 初始化训练监控器
    monitor_dir = work_root / "monitor"
    ensure_dir(monitor_dir)
    monitor = TrainingMonitor(monitor_dir)
    
    # 初始化模型（使用初始标注集训练）
    print("\n[初始化] 使用初始标注集训练基础模型...")
    init_save_dir = work_root / "round_0_initial"
    ensure_dir(init_save_dir)
    
    success, init_metrics = train_unet(
        round_num=0,
        labeled_img_dir=labeled_img_dir,
        labeled_mask_dir=labeled_mask_dir,
        pseudo_img_dir=pseudo_img_dir,  # 初始为空
        pseudo_mask_dir=pseudo_mask_dir,
        test_img_dir=test_img_dir,
        test_mask_dir=test_mask_dir,
        save_dir=init_save_dir,
        unet_script=unet_script,
        args=args
    )
    
    if not success:
        raise RuntimeError("初始模型训练失败！")
    
    best_model_path = init_save_dir / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError("初始模型文件不存在！")
    
    # 加载模型
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    print(f"[初始化] 基础模型训练完成 | Dice: {init_metrics.get('dice', 0.0):.4f}")
    
    # ==================== 主循环 ====================
    rounds_summary = []
    best_dice = init_metrics.get('dice', 0.0)
    best_round = 0
    
    random.seed(args.seed)
    
    for round_num in range(1, args.max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{args.max_rounds}")
        print(f"{'='*60}")
        
        # ==================== 检查终止条件 ====================
        available_images = get_all_images(train_dataset_dir)
        if not available_images:
            print(f"[终止] Train_Dataset 已空，共完成 {round_num - 1} 轮迭代")
            break
        
        print(f"[状态] Train_Dataset 剩余: {len(available_images)} 张")
        
        # ==================== 固定数量抽样（每轮 N 张）====================
        random.seed(round_num)  # 保证可复现
        sample_size = min(args.samples_per_round, len(available_images))
        sampled_images = random.sample(available_images, sample_size)
        print(f"[抽样] 本轮抽取: {len(sampled_images)} 张（固定={args.samples_per_round}）")
        
        # ==================== 预测 ====================
        round_work_dir = work_root / f"round_{round_num}"
        ensure_dir(round_work_dir)
        
        predictions = predict_batch(
            model=model,
            img_dir=train_dataset_dir,
            img_list=sampled_images,
            output_dir=round_work_dir,
            device=device
        )
        
        # ==================== LLM 评分 ====================
        scores_json = round_work_dir / "scores.json"
        scores = batch_evaluate_with_kimi(
            image_dir=train_dataset_dir,
            predictions=predictions,
            api_key=args.api_key,
            output_json=scores_json,
            suspicious_root=suspicious_root
        )
        
        if not scores:
            print(f"[警告] Round {round_num} LLM 评分失败，跳过本轮")
            remove_sampled_images(train_dataset_dir, sampled_images)
            continue
        
        # ==================== 样本选择 ====================
        selected_samples = select_high_confidence_samples(
            scores=scores,
            predictions=predictions,
            image_dir=train_dataset_dir,
            threshold=args.high_confidence
        )
        
        avg_score = np.mean([s['score'] for s in scores])
        selected_count = len(selected_samples)
        high_conf_rate = selected_count / len(sampled_images) if len(sampled_images) > 0 else 0.0
        
        print(f"[选择] 平均置信分: {avg_score:.2f}/10.0 | 入选: {selected_count}/{len(sampled_images)} ({high_conf_rate:.1%})")
        
        # ==================== 加入伪标签集 ====================
        if selected_samples:
            pseudo_manager.add_samples(selected_samples)
        
        # ==================== 备份 ====================
        backup_round_data(
            round_num=round_num,
            sampled_images=sampled_images,
            image_dir=train_dataset_dir,
            predictions=predictions,
            scores=scores,
            selected_samples=selected_samples,
            backup_root=backup_root
        )
        
        # ==================== 删除已抽取图像 ====================
        remove_sampled_images(train_dataset_dir, sampled_images)
        
        # ==================== 训练新模型（热启动）====================
        round_save_dir = work_root / f"round_{round_num}_model"
        ensure_dir(round_save_dir)
        
        # 热启动：从上一轮最佳模型继续
        pretrained = work_root / "best_model.pth" if (work_root / "best_model.pth").exists() else best_model_path
        
        success, metrics = train_unet(
            round_num=round_num,
            labeled_img_dir=labeled_img_dir,
            labeled_mask_dir=labeled_mask_dir,
            pseudo_img_dir=pseudo_img_dir,
            pseudo_mask_dir=pseudo_mask_dir,
            test_img_dir=test_img_dir,
            test_mask_dir=test_mask_dir,
            save_dir=round_save_dir,
            unet_script=unet_script,
            args=args,
            pretrained_model=pretrained
        )
        
        if not success:
            print(f"[警告] Round {round_num} 训练失败，跳过")
            continue
        
        # ==================== 更新最佳模型 ====================
        current_dice = metrics.get('dice', 0.0)
        print(f"[评估] Round {round_num} | Dice: {current_dice:.4f} | 最佳: {best_dice:.4f} (Round {best_round})")
        
        if current_dice > best_dice:
            best_dice = current_dice
            best_round = round_num
            
            # 保存最佳模型
            best_model_dst = work_root / "best_model.pth"
            shutil.copy(round_save_dir / "best_model.pth", best_model_dst)
            print(f"[更新] 最佳模型已更新 → Round {round_num} (Dice: {best_dice:.4f})")
        
        # ==================== 加载新模型 ====================
        new_model_path = round_save_dir / "best_model.pth"
        if new_model_path.exists():
            model.load_state_dict(torch.load(new_model_path, map_location=device))
        
        # ==================== 记录摘要 ====================
        round_summary = {
            "round": round_num,
            "sampled_count": len(sampled_images),
            "selected": selected_count,
            "avg_score_this_round": round(avg_score, 2),
            "high_confidence_rate": round(high_conf_rate, 4),
            "pseudo_total": pseudo_manager.get_current_count(),
            "dice": round(current_dice, 4),
            "precision": round(metrics.get('precision', 0.0), 4),
            "recall": round(metrics.get('recall', 0.0), 4),
            "f1": round(metrics.get('f1', 0.0), 4),
            "is_best": current_dice >= best_dice
        }
        rounds_summary.append(round_summary)
        
        # 记录到监控器
        monitor.record_round_metrics(round_num, {
            "sampled_count": len(sampled_images),
            "selected": selected_count,
            "avg_confidence": round(avg_score, 2),
            "high_confidence_rate": round(high_conf_rate, 4),
            "pseudo_total": pseudo_manager.get_current_count(),
            "dice": round(current_dice, 4),
            "precision": round(metrics.get('precision', 0.0), 4),
            "recall": round(metrics.get('recall', 0.0), 4),
            "f1": round(metrics.get('f1', 0.0), 4),
            "is_best": current_dice >= best_dice
        })
        
        # 更新当前状态
        monitor.update_status({
            "current_round": round_num,
            "max_rounds": args.max_rounds,
            "best_dice": best_dice,
            "best_round": best_round,
            "remaining_images": len(get_all_images(train_dataset_dir)),
            "pseudo_count": pseudo_manager.get_current_count(),
            "last_update": datetime.now().isoformat()
        })
        
        # 保存中间结果
        summary_path = work_root / "rounds_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(rounds_summary, f, indent=2, ensure_ascii=False)
    
    # ==================== 最终总结 ====================
    print(f"\n{'='*60}")
    print(f"滚动抽样自演化完成！")
    print(f"{'='*60}")
    print(f"总轮次: {len(rounds_summary)}")
    print(f"最佳 Dice: {best_dice:.4f} (Round {best_round})")
    print(f"最终伪标签数量: {pseudo_manager.get_current_count()}")
    print(f"最佳模型: {work_root / 'best_model.pth'}")
    print(f"轮次摘要: {work_root / 'rounds_summary.json'}")
    print(f"备份目录: {backup_root}")
    if suspicious_root:
        print(f"可疑样本: {suspicious_root}")
    print(f"{'='*60}\n")
    
    # 保存最终摘要
    final_summary = {
        "total_rounds": len(rounds_summary),
        "best_round": best_round,
        "best_dice": best_dice,
        "final_pseudo_count": pseudo_manager.get_current_count(),
        "rounds": rounds_summary
    }
    
    final_path = work_root / "final_summary.json"
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)


# ==================== CLI ====================
def parse_args() -> argparse.Namespace:
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description="医学图像分割自演化系统（固定每轮30张抽样）")
    
    # 数据路径
    parser.add_argument("--train-dataset-dir", type=str, required=True, help="待标注图像池目录")
    parser.add_argument("--labeled-img-dir", type=str, required=True, help="初始标注图像目录")
    parser.add_argument("--labeled-mask-dir", type=str, required=True, help="初始标注掩码目录")
    parser.add_argument("--test-img-dir", type=str, required=True, help="测试集图像目录")
    parser.add_argument("--test-mask-dir", type=str, required=True, help="测试集掩码目录")
    
    # 工作路径
    parser.add_argument("--work-root", type=str, default="./work", help="工作根目录")
    parser.add_argument("--backup-root", type=str, default="./backup", help="备份根目录")
    parser.add_argument("--suspicious-root", type=str, default=None, help="可疑高分样本目录（可选）")
    parser.add_argument("--unet-script", type=str, default="./unet", help="UNet 训练脚本路径")
    
    # API 配置
    parser.add_argument("--api-key", type=str, required=True, help="Kimi API Key")
    
    # 核心参数
    parser.add_argument("--samples-per-round", type=int, default=30, help="每轮固定抽样数量")
    parser.add_argument("--high-confidence", type=float, default=7.5, help="高置信度阈值（0-10）")
    parser.add_argument("--max-rounds", type=int, default=50, help="最大迭代轮次")
    parser.add_argument("--max-pseudo-samples", type=int, default=5000, help="伪标签集最大容量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--epoch", type=int, default=10, help="每轮训练轮数")
    parser.add_argument("--dice-weight", type=float, default=0.6, help="Dice Loss 权重")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--use-amp", action="store_true", help="启用混合精度训练")
    
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    run_rolling_evolution(args)


if __name__ == "__main__":
    main()
