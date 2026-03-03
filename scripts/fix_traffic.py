import pandas as pd
import os

# 1. 检查文件是否存在
input_path = './dataset/traffic.csv'
output_path = './dataset/traffic_fixed.csv'

if not os.path.exists(input_path):
    print(f"❌ 错误：在 {input_path} 没找到文件！请确认你把 traffic.csv 复制到 dataset 文件夹里了吗？")
else:
    print(f"正在读取 {input_path} (数据量大，可能需要 1-2 分钟，请耐心等待)...")

    # 2. 读取数据
    df_long = pd.read_csv(input_path)

    # 3. 检查格式
    if 'cols' in df_long.columns and 'data' in df_long.columns:
        print("格式正确 (长格式)，开始转换 (Pivoting)... ⚠️ 内存会飙升，千万别关窗口！")

        # 4. 转换：把 cols 列的内容变成表头
        df_wide = df_long.pivot(index='date', columns='cols', values='data')

        # 5. 整理
        df_wide.reset_index(inplace=True)
        df_wide.columns.name = None

        # 6. 保存
        print("转换完成，正在保存...")
        df_wide.to_csv(output_path, index=False)
        print(f"✅ 成功！已保存为: {output_path}")
        print(f"最终特征数 (列数): {df_wide.shape[1]-1} (预期是 862)")
    else:
        print("❌ 格式不对！列名必须包含 'date', 'data', 'cols'")
