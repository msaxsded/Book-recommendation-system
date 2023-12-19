import streamlit as st  
import pandas as pd  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
  
# 加载数据集  
data = pd.read_csv('book_ratings.csv')  
  
# 提取特征  
book_features = data['book'].values  
book_features = [f'{book} {features}' for book, features in zip(data['user'], book_features)]  
  
# 创建CountVectorizer对象  
vectorizer = CountVectorizer()  
  
# 将特征转换为向量形式  
book_vectors = vectorizer.fit_transform(book_features)  
  
# 计算相似度矩阵  
similarity_matrix = cosine_similarity(book_vectors)  
  
# 构建推荐函数  
def recommend(user):  
    # 获取用户已评分的书籍列表  
    rated_books = data[data['user'] == user]['book'].values  
    # 获取用户未评分的书籍列表  
    unrated_books = set(data['book'].unique()) - set(rated_books)  
    # 获取与用户已评分书籍最相似的书籍列表  
    similar_books = [books[0] for books in similarity_matrix[user] if books[0] in unrated_books]  
    # 返回推荐结果  
    return similar_books[::-1]  # 按相似度降序排列  
  
# 展示推荐结果  
st.write('Book Recommendation System')  
st.write('===========================')  
st.write('Enter a user ID to see their recommended books.')  
st.text_input('User ID', '')  
user = st.text_input('Enter a user ID:', '') or '1'  # 默认第一个用户为示例  
recommended_books = recommend(user)  
st.write('Recommended books for user:', user)  
st.write('===========================')  
st.write(recommended_books)
