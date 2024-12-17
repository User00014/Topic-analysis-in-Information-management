import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 读取CSV文件
file_path = r"C:\Users\13488\Desktop\Web数据分析实验指导书\期末论文\analysis\bigmodel.csv"
data = pd.read_csv(file_path)

# 检查是否存在Method和Theory列
if 'Method' in data.columns and 'Theory' in data.columns:
    # 提取研究方法和理论
    methods = data['Method'].dropna()  # 删除缺失值
    theories = data['Theory'].dropna()  # 删除缺失值
else:
    print("CSV file does not contain 'Method' or 'Theory' columns.")
    exit()

# NLP预处理
def preprocess_text(text):
    # 分词
    tokens = word_tokenize(text)
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return " ".join(filtered_tokens)

# 应用预处理
methods_processed = methods.apply(preprocess_text)
theories_processed = theories.apply(preprocess_text)

# 统计研究方法和理论的出现次数
method_counts = methods_processed.apply(lambda x: x.split()).explode().value_counts()
theory_counts = theories_processed.apply(lambda x: x.split()).explode().value_counts()

# 过滤出现次数大于1的项
method_counts_filtered = method_counts[method_counts > 1]
theory_counts_filtered = theory_counts[theory_counts > 1]

# 可视化研究方法的变化趋势
plt.figure(figsize=(10, 6))
sns.barplot(x=method_counts_filtered.index, y=method_counts_filtered.values)
plt.title('Research Method Trend (Occurrences > 1)')
plt.xlabel('Research Method')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 可视化理论的变化趋势
plt.figure(figsize=(10, 6))
sns.barplot(x=theory_counts_filtered.index, y=theory_counts_filtered.values)
plt.title('Theory Trend (Occurrences > 1)')
plt.xlabel('Theory')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()