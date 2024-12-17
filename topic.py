import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# 尝试加载NLTK的停用词和WordNet资源
try:
    _stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    print("NLTK resources not found. Please download them using nltk.download() or set the correct path.")
    _stop_words = set()
    lemmatizer = None

# 读取CSV文件，使用ISO-8859-1编码格式
file_path = "C:\\Users\\13488\\Desktop\\Web数据分析实验指导书\\期末论文\\analysis\\journal_abstracts.csv"
try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
except Exception as e:
    print(f"Failed to read the CSV file: {e}")
    exit()

# 数据预处理
def preprocess(text):
    if pd.isnull(text) or isinstance(text, float) and not isinstance(text, bool):
        return [], None  # 返回空列表和None，表示无法处理
    text = str(text).lower()
    words = [w for w in text.split() if w not in _stop_words and w.isalpha()]
    words = [lemmatizer.lemmatize(w) for w in words if lemmatizer]
    return words, ' '.join(words)  # 返回处理后的单词列表和空格分隔的字符串

# 提取摘要和发布日期
abstracts = df['abstract']
dates = df['published_date']

# 将日期字符串转换为日期对象
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

# 预处理文本，并跳过无法处理的数据
processed_abstracts = []
for text in tqdm(abstracts, desc='Preprocessing abstracts'):
    result, processed_text = preprocess(text)
    if result:  # 如果结果不为空，添加到列表中
        processed_abstracts.append(processed_text)

# 使用TF-IDF模型
if not processed_abstracts:
    print("No abstracts were processed. Exiting.")
    exit()
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(processed_abstracts)

# 主题建模
lda_model = LatentDirichletAllocation(n_components=5, random_state=0)
lda_model.fit(tfidf)

# 获取每个文档的主题分布
topic_distribution = lda_model.transform(tfidf)

def get_topic_keywords(topic, feature_names, no_words=10):
    # 获取每个词的索引和权重
    topic_terms = [(feature_names[i], topic[i]) for i in range(len(topic))]
    # 按权重排序
    topic_terms.sort(key=lambda x: x[1], reverse=True)
    # 返回权重最高的前no_words个词
    return [term[0] for term in topic_terms[:no_words]]

# 获取主题关键词
topic_keywords = []
no_words = 10
feature_names = tfidf_vectorizer.get_feature_names_out()  # 确保已经获取了特征名称
for topic in lda_model.components_:
    topic_keywords.append(get_topic_keywords(topic, feature_names, no_words))

# 打印每个主题的关键词
for i, keywords in enumerate(topic_keywords):
    print(f"Topic {i+1}: {keywords}")

# 主题随时间变化的关系图
time_distribution = {i: [] for i in range(lda_model.n_components)}
for i, doc in tqdm(enumerate(topic_distribution), total=len(topic_distribution), desc='Processing topics over time'):
    dominant_topic = np.argmax(doc)
    year = df.loc[i, 'published_date'].year if pd.notnull(df.loc[i, 'published_date']) else None
    if year:
        time_distribution[dominant_topic].append(year)

# 绘制主题随时间变化的图
plt.figure(figsize=(12, 6))

# 为每个主题创建一个列表来存储每年的文档数量
topic_year_counts = {i: [] for i in range(lda_model.n_components)}
for i, doc in tqdm(enumerate(topic_distribution), total=len(topic_distribution), desc='Processing topics over time'):
    dominant_topic = np.argmax(doc)
    year = df.loc[i, 'published_date'].year if pd.notnull(df.loc[i, 'published_date']) else None
    if year:
        topic_year_counts[dominant_topic].append(year)

# 准备数据框，用于绘制图形
data_for_plot = []
for i in range(lda_model.n_components):
    for year in set(topic_year_counts[i]):  # 使用set去重
        count = topic_year_counts[i].count(year)
        data_for_plot.append({'Year': year, 'Topic': i+1, 'Count': count})

df_plot = pd.DataFrame(data_for_plot)

# 绘制每个主题的直方图
sns.barplot(data=df_plot, x='Year', y='Count', hue='Topic', palette='viridis')

plt.title('Topic Distribution Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Documents')

# 为图例设置标题
plt.legend(title='Topic Number', labels=[f'Topic {i+1}: {" / ".join(topic_keywords[i])}' for i in range(lda_model.n_components)])

# 调整布局以防止图例被遮挡
plt.tight_layout()

plt.show()