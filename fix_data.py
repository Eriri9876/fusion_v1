import pandas as pd

# 1. 读取你那个“竖着”的文件
print("正在读取 CSV 文件 (可能需要几秒钟)...")
df_long = pd.read_csv('./dataset/ETTh1.csv')

# 2. 检查一下是不是我们想的那样
if 'cols' in df_long.columns and 'data' in df_long.columns:
    print("识别到长格式数据，正在进行透视转换 (Pivoting)...")

    # 3. 关键一步：把 'cols' 列里的内容变成表头
    df_wide = df_long.pivot(index='date', columns='cols', values='data')

    # 4. 把 date 索引变回一列
    df_wide.reset_index(inplace=True)

    # 5. 去掉列名的层级名字
    df_wide.columns.name = None

    # 6. 保存为新文件
    output_path = './dataset/ETTh1_fixed.csv'
    df_wide.to_csv(output_path, index=False)
    print(f"✅ 转换成功！已保存为: {output_path}")
    print("现在的列名是:", df_wide.columns.tolist())
else:
    print("❌ 格式看起来不对，请确认文件里有 'cols' 和 'data' 列")