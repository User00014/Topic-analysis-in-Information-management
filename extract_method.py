import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 指定本地BERT模型的路径
local_bert_model_path = "C:\\Users\\13488\\Desktop\\大创\\大创用\\数据\\bert-base-uncased\\bert-base-uncased"

# 加载本地预训练的BERT模型和分词器
tokenizer = AutoTokenizer.from_pretrained(local_bert_model_path)
model = AutoModel.from_pretrained(local_bert_model_path)

# 读取CSV文件
file_path = "C:\\Users\\13488\\Desktop\\Web数据分析实验指导书\\期末论文\\analysis\\journal_abstracts.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 预处理文本数据
def preprocess(texts):
    return tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

# 应用预处理到摘要字段
abstracts = df['abstract'].dropna().tolist()  # 确保移除空值
inputs = preprocess(abstracts)

# 使用BERT模型提取特征
with torch.no_grad():
    outputs = model(**inputs)

# 假设我们有一个函数来识别研究方法和理论
def extract_methods_theories(features):
    # 这里应该是一个复杂的NLP任务，可能涉及到命名实体识别或其他技术
    # 为了示例，我们返回一些模拟数据
    return ["Method1", "Theory1", "Method2", "Theory2"]

# 提取研究方法和理论
methods_theories = [extract_methods_theories(output.last_hidden_state[0]) for output in outputs]

# 可视化分析
# 这里我们使用模拟数据，实际应用中需要根据extract_methods_theories函数的输出进行调整
methods_counts = {method: methods_theories.count(method) for method in set(methods_theories)}
plt.figure(figsize=(10, 6))
sns.barplot(x=list(methods_counts.keys()), y=list(methods_counts.values()))
plt.title('Research Methods and Theories Distribution')
plt.xlabel('Methods/Theories')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()