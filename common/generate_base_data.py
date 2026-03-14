#!/usr/bin/env python3
"""
一键生成基础数据脚本
执行此脚本生成所有实验共用的基础数据，大幅加速后续训练

使用方法:
    python scripts/generate_base_data.py
    python scripts/generate_base_data.py --max_users 10  # 快速测试
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from common import BaseGeoLifePreprocessor


def main():
    parser = argparse.ArgumentParser(
        description='GeoLife基础数据预处理（所有实验通用）'
    )
    parser.add_argument(
        '--geolife_root',
        type=str,
        default='../data/Geolife Trajectories 1.3',
        help='GeoLife数据根目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/processed/base_segments.pkl',
        help='输出缓存路径'
    )
    parser.add_argument(
        '--max_users',
        type=int,
        default=None,
        help='最大用户数（用于快速测试，留空处理全部）'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=10,
        help='最小轨迹段长度'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("GeoLife 基础数据预处理")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  数据根目录: {args.geolife_root}")
    print(f"  输出路径: {args.output}")
    print(f"  最大用户数: {args.max_users or '全部'}")
    print(f"  最小长度: {args.min_length}")

    # 检查数据目录
    if not os.path.exists(args.geolife_root):
        print(f"\n❌ 错误: 找不到GeoLife数据目录: {args.geolife_root}")
        print("\n请确保:")
        print("  1. 已下载 GeoLife Trajectories 1.3 数据集")
        print("  2. 数据路径正确")
        return

    # 创建预处理器
    preprocessor = BaseGeoLifePreprocessor(args.geolife_root)

    # 处理数据
    print("\n开始处理...")
    segments = preprocessor.process_all_users(
        max_users=args.max_users,
        min_segment_length=args.min_length
    )

    if not segments:
        print("\n❌ 错误: 未提取到任何轨迹段")
        return

    # 保存缓存
    preprocessor.save_to_cache(segments, args.output)

    print("\n" + "=" * 80)
    print("✅ 基础数据预处理完成！")
    print("=" * 80)

    print("\n📊 数据文件:")
    print(f"  路径: {args.output}")
    print(f"  大小: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")
    print(f"  轨迹段数: {len(segments)}")

    print("\n🚀 后续使用:")
    print("  现在可以直接运行各实验的训练脚本，无需重复处理GeoLife数据：")
    print("  ")
    print("  # Exp1")
    print("  cd exp1")
    print("  python train.py --use_base_data")
    print("  ")
    print("  # Exp2")
    print("  cd exp2")
    print("  python train.py --use_base_data")
    print("  ")
    print("  # Exp3")
    print("  cd exp3")
    print("  python train.py --use_base_data")
    print("  ")
    print("  # Exp4")
    print("  cd exp4")
    print("  python train.py --use_base_data")

    print("\n⏱️  预计时间节省:")
    print("  首次运行: 30-60分钟（生成基础数据）")
    print("  后续训练: 节省 80-90% 的数据处理时间")


if __name__ == '__main__':
    main()