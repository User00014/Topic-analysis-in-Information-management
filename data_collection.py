# import pandas as pd
# import os
# import re
#
# # 指定文件夹路径
# folder_path = "C:/Users/13488/Desktop/Web数据分析实验指导书/期末论文/期刊内容摘要"
#
# # 初始化一个空的DataFrame，用于存储合并后的数据
# combined_df = pd.DataFrame()
#
# # 定义一个函数来提取年份
# def extract_year(date):
#     if pd.isnull(date):
#         return date  # 如果是空值，则保持不变
#     match = re.search(r'\b(19|20)\d{2}\b', str(date))
#     if match:
#         return int(match.group())  # 提取并返回年份
#     else:
#         return date  # 如果没有匹配到年份，则保持原值
#
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):  # 确保处理的是CSV文件
#         file_path = os.path.join(folder_path, filename)
#
#         # 读取CSV文件，从第二列开始读取
#         df = pd.read_csv(file_path, usecols=lambda column: column != '1')
#
#         # 数据清洗
#         # 移除source字段中的后缀'arrow_drop_down'
#         if 'source' in df.columns:
#             df['source'] = df['source'].str.replace('arrow_drop_down', '', regex=False)
#
#         # 统一published_date格式为年份
#         if 'published_date' in df.columns:
#             df['published_date'] = df['published_date'].apply(
#                 lambda x: extract_year(x)
#             )
#
#         # 处理citation列，如果不存在则添加，并统一为-1
#         if 'citation' not in df.columns:
#             df['citation'] = -1
#         else:
#             df['citation'] = df['citation'].fillna(-1)  # 将NaN值替换为-1
#
#         # 合并数据
#         combined_df = pd.concat([combined_df, df], ignore_index=True)
#
# # 删除合并后DataFrame的第一列
# if combined_df.shape[1] > 1:  # 确保DataFrame不为空且有超过一列
#     combined_df = combined_df.drop(combined_df.columns[0], axis=1)
#
# # 保存合并后的DataFrame到新的CSV文件
# combined_df.to_csv("combined_journal_abstracts.csv", index=False)


import pandas as pd

df = pd.read_csv('combined_journal_abstracts.csv')
df = df.dropna(subset=['source'])
df.tocsv('combined_journal_abstracts.csv', index=False)
