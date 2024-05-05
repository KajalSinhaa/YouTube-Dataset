#!/usr/bin/env python
# coding: utf-8

# #                                  EDA on Youtube Dataset

# ### 1. Import required libraries and read the provided dataset (youtube_dislike_dataset.csv) and retrieve top 5 and bottom 5 records.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv(r"downloads/youtube_dislike_dataset.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# ### 2. Check the info of the dataframe and write your inferences on data types and shape of the dataset.

# In[5]:


df.info()


# In[6]:


df.shape


# The Following dataframe consists of 37422 rows and 12 columns. Each row represents the data about a video published on youtube. The dataframe have 12 variables and is having Null Values only in the comments column. The dataframe have 8 columns of object datatypes and 4 of Int64 datatypes. The memory used to save this dataframe is 3.4+ MB.

# ### 3. Check for the Percentage of the missing values and drop or impute them.

# In[8]:


df.isnull().mean() * 100


# In[9]:


df.drop(columns="comments",inplace= True)


# In[10]:


df.isnull().sum()


# ### 4. Check the statistical summary of both numerical and categorical columns and write your inferences.

# In[12]:


df.describe()


# In[13]:


df.describe(include=["object"])


# The Numerical Columns consists of view_counts, likes, dislikes, comment_count. The Statistical summary for the numerical datatypes provides us with the descriptive details about our dataset. We can get a good idea from viewing the mean of the numberical columns of the dataframe and getting the quartiles helps us to undersatand the data points. Whereas the Categorical column provides us with a good understanding of the dataset as a whole.
# 

# ### 5. Convert datatype of column published_at from object to pandas datetime.

# In[14]:


df['published_at'] = pd.to_datetime(df['published_at'])


# In[15]:


df.dtypes


# ### 6. Create a new column as 'published_month' using the column published_at (display the months only).

# In[16]:


df['published_month'] = df['published_at'].dt.month


# In[17]:


df


# ### 7. Replace the numbers in the column published_month as names of the months i,e., 1 as 'Jan', 2 as 'Feb' and so on.....

# In[18]:


month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 
               8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df['published_month'] = df['published_month'].map(month_names)


# In[19]:


df


# ### 8. Find the number of videos published each month and arrange the months in a decreasing order based on the video count.

# In[20]:


video_count_per_month = df['published_month'].value_counts()


# In[21]:


video_count_per_month = video_count_per_month.sort_values(ascending=False)


# In[22]:


video_count_per_month


# ### 9. Find the count of unique video_id, channel_id and channel_title.

# In[24]:


unique_video_id_count = df['video_id'].nunique()
unique_channel_id_count = df['channel_id'].nunique()
unique_channel_title_count = df['channel_title'].nunique()

print("Unique video_id count:", unique_video_id_count)
print("Unique channel_id count:", unique_channel_id_count)
print("Unique channel title count:", unique_channel_title_count)


# ### 10. Find the top10 channel names having the highest number of videos in the dataset and the bottom10 having lowest number of videos.

# In[25]:


channel_video_count = df['channel_title'].value_counts()


# In[26]:


top_10_channels = channel_video_count.head(10)


# In[27]:


top_10_channels


# In[28]:


bottom_10_channels = channel_video_count.tail(10)
bottom_10_channels


# ### 11. Find the title of the video which has the maximum number of likes and the title of the video having minimum likes and write your inferences.

# In[29]:


max_likes_title = df.loc[df['likes'].idxmax(), 'title']
max_likes_title 


# In[30]:


min_likes_title = df.loc[df['likes'].idxmin(), 'title']
min_likes_title


# The most liked video is the "BTS () 'Dynamite' Official MV" and the minimum is of "Kim Kardashian\'s Must-See Moments on "Saturday Night Live"| E! News".

# ### 12. Find the title of the video which has the maximum number of dislikes and the title of the video having minimum dislikes and write your inferences.

# In[31]:


max_dislikes_title = df.loc[df['dislikes'].idxmax(), 'title']
max_dislikes_title


# In[32]:


min_dislikes_title = df.loc[df['dislikes'].idxmin(), 'title']
min_dislikes_title


# The least dislikes video is: \'Kim Kardashian\'s Must-See Moments on "Saturday Night Live" | E! News and the most disliked video is \'Cuties | Official Trailer | Netflix\'.It is also observed that the videos having mimimun number of likes are having mimimum number of dislikes too.

# ### 13. Does the number of views have any effect on how many people disliked the video? Support your answer with a metric and a plot.

# In[44]:


correlation = df['view_count'].corr(df['dislikes'])


# In[45]:


correlation


# In[46]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='view_count', y='dislikes', data=df)
plt.title(f'Scatter Plot of view_count vs dislikes')
plt.xlabel('view_count')
plt.ylabel('dislikes')
plt.grid(True)
plt.show()


# The correlation between views and dislikes is 0.684. It means it's having a direct relation and increase in one leads to increase in another and vice-versa.
# 
# 

# ### 14. Display all the information about the videos that were published in January, and mention the count of videos that were published in January.

# In[34]:


january_videos = df[df['published_month'] == 'Jan']


# In[35]:


january_videos


# In[36]:


january_videos.describe()


# In[37]:


january_videos_count = len(january_videos)


# In[38]:


january_videos_count

