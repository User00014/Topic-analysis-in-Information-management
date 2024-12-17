import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm

# 确保NLTK资源已下载
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# 读取CSV文件
file_path = "C:\\Users\\13488\\Desktop\\Web数据分析实验指导书\\期末论文\\analysis\\journal_abstracts.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 数据预处理
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words and word.isalpha()]
    return " ".join(words)

# 应用预处理到摘要字段，并显示进度条
df['title'] = df['title'].apply(preprocess)
df['title'] = df['title'].dropna()  # 删除空值

# 使用TF-IDF模型
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
tfidf = tfidf_vectorizer.fit_transform(df['title'])


# 主题建模
lda_model = LatentDirichletAllocation(n_components=5, random_state=0)
lda_model.fit(tfidf)

# 获取文档-主题矩阵
doc_topic_matrix = lda_model.transform(tfidf)

# 将主题信息添加到 DataFrame
df['Dominant Topic'] = doc_topic_matrix.argmax(axis=1)

# 统计每个机构在每个主题下的文档数量
topic_affiliation_counts = df.groupby(['Dominant Topic', 'affiliations']).size().unstack(fill_value=0)

# 绘制散点图
fig, ax = plt.subplots(figsize=(12, 8))
affiliations = topic_affiliation_counts.columns
topics = topic_affiliation_counts.index

for i, topic in enumerate(topics):
    for j, affiliation in enumerate(affiliations):
        ax.scatter(i, j, s=topic_affiliation_counts.loc[topic, affiliation]*10, alpha=0.5, label=affiliation if topic_affiliation_counts.loc[topic, affiliation] > 0 else "")

# 设置图表属性
ax.set_title('Scatter Plot of Topics vs Affiliations')
ax.set_xlabel('Topic Number')
ax.set_ylabel('Affiliation Index')
ax.set_yticks(range(len(affiliations)))
ax.set_yticklabels(affiliations)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., labelspacing=1)
plt.show()