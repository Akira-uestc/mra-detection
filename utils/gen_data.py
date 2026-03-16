import pyreadr
import pandas as pd
import numpy as np

# 文件路径配置
NORMAL_DATA_PATH = 'dataset/TEP_FaultFree_Training.RData'
FAULTY_DATA_PATH = 'dataset/TEP_Faulty_Training.RData'
NORMAL_COUNT = 1500
FAULTY_COUNT = 1500
TOTAL_STEP = NORMAL_COUNT + FAULTY_COUNT # 共 3000

def process_and_export():
    try:
        print("正在按前后块状分布处理数据，请稍等...")
        normal_res = pyreadr.read_r(NORMAL_DATA_PATH)
        faulty_res = pyreadr.read_r(FAULTY_DATA_PATH)
        
        # 1. 分别提取前1500行正常数据和前1500行故障数据
        df_normal_part = normal_res[list(normal_res.keys())[0]].head(NORMAL_COUNT).copy()
        df_faulty_part = faulty_res[list(faulty_res.keys())[0]].head(FAULTY_COUNT).copy()

        # 2. 纵向合并：前 1500 正常，后 1500 故障
        # ignore_index=True 可以让索引变成 0-2999
        combined_df = pd.concat([df_normal_part, df_faulty_part], axis=0, ignore_index=True)

        # 3. 模拟多速率采样逻辑
        cols = list(combined_df.columns)
        
        for col in cols:
            if col.startswith('xmeas_'):
                try:
                    m_idx = int(col.split('_')[1])
                    
                    # 第一组：1-13 (全采样)
                    if 1 <= m_idx <= 13:
                        continue 
                    
                    # 第二组：14-27 (1/3 采样)
                    elif 14 <= m_idx <= 27:
                        interval = 3
                        mask = np.arange(len(combined_df)) % interval != 0
                        combined_df.loc[mask, col] = np.nan
                        
                    # 第三组：28-41 (1/6 采样)
                    elif 28 <= m_idx <= 41:
                        interval = 6
                        mask = np.arange(len(combined_df)) % interval != 0
                        combined_df.loc[mask, col] = np.nan
                except (ValueError, IndexError):
                    continue
            
            # xmv 变量保持不变，全采样
            elif col.startswith('xmv_'):
                continue

        # 4. 导出 CSV
        output_file = 'TEP_3000_Block_Split.csv'
        combined_df.to_csv(output_file, index=False)
        
        # 5. 打印验证信息
        print("-" * 35)
        print(f"处理完成！数据已按块划分")
        print(f"数据总行数: {len(combined_df)}")
        print(f"正常数据区间: 0 - {NORMAL_COUNT-1}")
        print(f"故障数据区间: {NORMAL_COUNT} - {TOTAL_STEP-1}")
        print(f"xmeas_14 (1/3采样) 非空数: {combined_df['xmeas_14'].count()}")
        print(f"xmeas_28 (1/6采样) 非空数: {combined_df['xmeas_28'].count()}")
        print(f"xmv_1    (全采样)  非空数: {combined_df['xmv_1'].count()}")
        print(f"文件已保存至: {output_file}")

    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    process_and_export()