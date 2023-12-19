import streamlit as st  
import pandas as pd  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
  
# 加载数据集  
books_data = pd.read_csv('Books.csv')   # 包含书籍描述信息的文件  
ratings_data = pd.read_csv('Ratings.csv')  # 包含用户评分信息的文件  
users_data = pd.read_csv('Users.csv')   # 包含用户信息的文件  
  
# 提取特征：从书籍描述中提取特征  
book_features = books_data['ISBN'].values  # 假设书籍描述存储在'description'列中  
  
# 创建CountVectorizer对象  
vectorizer = CountVectorizer()  
  
# 将特征转换为向量形式  
book_vectors = vectorizer.fit_transform(book_features)  
  
# 计算相似度矩阵  
similarity_matrix = cosine_similarity(book_vectors)  
  
# 构建推荐函数  
def recommend(user):  
    # 获取用户已评分的书籍列表  
    rated_books = ratings_data[ratings_data['user'] == user]['book'].values  
    # 获取用户未评分的书籍列表  
    unrated_books = set(books_data['book'].unique()) - set(rated_books)  
    # 获取与用户已评分书籍最相似的书籍列表  
    similar_books = [books[0] for books in similarity_matrix[user] if books[0] in unrated_books]  
    # 返回推荐结果，按相似度降序排列  
    return similar_books[::-1]  # 按相似度降序排列  
  
# 展示推荐结果  
st.write('Book Recommendation System')  
st.write('===========================')  
st.write('Enter a user ID to see their recommended books.')  
user_id = st.text_input('User ID', '') or '1'  # 默认第一个用户为示例  
recommended_books = recommend(user_id)  # 将用户ID传递给推荐函数  
st.write('Recommended books for user:', user_id)  
st.write('===========================')  
st.write(recommended_books) if recommended_books else st.write('No recommended books for this user.')
