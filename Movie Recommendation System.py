#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Here in this Movie Recommendation System when a user enters a name of the movie, this model provides a list of similar movies
# While building this system we will build a mode based on content based recommendation and popularity based recommendation system
# Here we use cosine similarity which is used to find similarity between the vectors


# In[42]:


# We imported the dataset from Kaggle and now importing the required libraries
import pandas as pd
import numpy as np
import difflib # If there is any mistake in the entered username by the user this library helps to take the close match for that movie name
from sklearn.feature_extraction.text import TfidfVectorizer
# This library helps in converting the Textual data into corresponding feature vectors which are Numerical data
from sklearn.metrics.pairwise import cosine_similarity


# In[43]:


dataframe = pd.read_csv('movies.csv')
dataframe.head()


# In[44]:


dataframe.shape


# In[45]:


required_features = ['genres','keywords','original_title','popularity','tagline','cast','director']


# In[46]:


# Check whether we have any missing values 
dataframe.isnull().sum()


# In[47]:


# Since we have missing values in some rows we have to replace with null
dataframe = dataframe.fillna('')
dataframe.isnull().sum()
#If we do not want to replace all the missing values in columns, and want to replace only for the selected features by using for loop
# for i in required_features:
#    dataframe[i] = dataframe[i].fillna('')


# In[48]:


dataframe['New_column'] = dataframe['genres'] + ' ' + dataframe['keywords'] + ' ' + dataframe['original_title'] + ' '+ str(dataframe['popularity']) + ' '+ str(dataframe['tagline']) + ' ' + dataframe['cast'] + ' ' + dataframe['director']


# In[49]:


Latest_Features = dataframe['New_column']


# In[50]:


Latest_Features.head()


# In[51]:


# Now converting this column into feature vectors which is into Numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(Latest_Features)
Latest_Features = vectorizer.transform(Latest_Features)


# In[52]:


print(Latest_Features)


# In[53]:


# Now getting the similarity score using Cosine Similarity
similarity = cosine_similarity(Latest_Features)
similarity


# In[54]:


similarity.shape
# Here after performing the cosine similarity we have 4803 rows and 4803 columns which represents if we enter a movie name how much that
# movie is similar compared to other 4802 movies can be noted by using cosine similarity


# In[89]:


# Now we have to take the input from the user which is the movie_name
movie_name = input("enter the movie_name: ")


# In[90]:


# Now instead of printing the entered movie it has to recommend some movies based on the cosine similarity of the entered movie
movie_names = dataframe['title'].tolist()
movie_names


# In[91]:


# Now finding the closest match for the movie name which was entered by the user
# So, take the input which was entered by the user and then compare with all the movie_names and return closest similarity movie_names using difflib library 
similar_name_movies = difflib.get_close_matches(movie_name,movie_names)
print(similar_name_movies)


# In[92]:


similar_name_movies = similar_name_movies[0]
print(similar_name_movies)


# In[93]:


movie_index = dataframe[dataframe.title == similar_name_movies]['index'].values[0]
print(movie_index)
# Finding the index 


# In[94]:


# Now since we got the index we go to that corresponding row in cosine similarity matrix and check the similarity movies
#similarity[movie_index]
similarity_score = list(enumerate(similarity[movie_index]))
print(similarity_score)


# In[95]:


# Here we want only the top 5 movies or top 10 movies which are similar 
# Sorting the similarity score
#similarity_score = sorted(similarity_score)
#print(similarity_score)


# In[96]:


sorted_similarity_score = sorted(similarity_score,key= lambda x:x[1],reverse=True)
print(sorted_similarity_score)


# In[97]:


# Now based on this index we have to print 
#movie_name = dataframe[dataframe.index(similarity_score,key=lambda x:x[0])]


# In[107]:


k=1
print("Based on your recent watched movie your recommended movies are:")
print(' ')
for i in sorted_similarity_score:
    index = i[0]
    movie_title = dataframe[dataframe.index==index]['title'].values[0]
    if k<21:
        print(k,' ',movie_title)
        k+=1


# In[ ]:


movie_name = input("enter the movie_name: ")
movie_names = dataframe['title'].tolist()
similar_name_movies = difflib.get_close_matches(movie_name,movie_names)
similar_name_movies = similar_name_movies[0]
movie_index = dataframe[dataframe.title == similar_name_movies]['index'].values[0]
similarity_score = list(enumerate(similarity[movie_index]))
sorted_similarity_score = sorted(similarity_score,key= lambda x:x[1],reverse=True)
k=1
print("Based on your recent watched movie your recommended movies are:")
print(' ')
for i in sorted_similarity_score:
    index = i[0]
    movie_title = dataframe[dataframe.index==index]['title'].values[0]
    if k<21:
        print(k,' ',movie_title)
        k+=1


# In[ ]:


movie_name = input("enter the movie_name: ")
movie_names = dataframe['title'].tolist()
similar_name_movies = difflib.get_close_matches(movie_name,movie_names)
similar_name_movies = similar_name_movies[0]
movie_index = dataframe[dataframe.title == similar_name_movies]['index'].values[0]
similarity_score = list(enumerate(similarity[movie_index]))
sorted_similarity_score = sorted(similarity_score,key= lambda x:x[1],reverse=True)
k=1
print("Based on your recent watched movie your recommended movies are:")
print(' ')
for i in sorted_similarity_score:
    index = i[0]
    movie_title = dataframe[dataframe.index==index]['title'].values[0]
    if k<21:
        print(k,' ',movie_title)
        k+=1


# In[ ]:




