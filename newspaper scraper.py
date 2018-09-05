
# coding: utf-8

# In[3]:


#!pip install newspaper3k


# In[2]:


import newspaper


# In[3]:


#newspaper.popular_urls()


# In[4]:


#print(len(newspaper.popular_urls()))


# In[44]:


#cnn_paper = newspaper.build('http://cnn.com')
nyt_paper = newspaper.build('http://nytimes.com')
#guardian_paper = newspaper.build('http://www.guardiannews.com')
#abc_paper = newspaper.build('http://abcnews.com')
#alja_paper = newspaper.build('http://www.aljazeera.com')
#sky_paper = newspaper.build('http://news.sky.com')

#papers = [cnn_paper, abc_paper, was_post_paper, alja_paper, sky_paper]


# In[45]:


for articles in nyt_paper.articles:
    print(articles.url)


# In[46]:


articles = nyt_paper.articles
text = []
for article in articles:
    try:
        article.download()
        article.parse()
        text.append(article.text)
    except:
        pass


# In[47]:


text


# In[48]:


with open('nyt_paper.txt', 'w') as file_handler:
    for item in text:
        try:
            file_handler.write(item)
        except:
            pass

