#Import dependencies
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import json


def get_blog_articles():

    article_list = ['https://codeup.com/codeup-news/dei-report/', 'https://codeup.com/codeup-news/diversity-and-inclusion-award/','https://codeup.com/featured/financing-career-transition/','https://codeup.com/tips-for-prospective-students/tips-for-women/','https://codeup.com/cloud-administration/cloud-computing-and-aws/']
    file = 'blog_posts.json'
    
    if os.path.exists(file):
        
        with open(file) as f:
        
            return json.load(f)
    
    headers = {'User-Agent': 'Codeup Data Science'}
    
    article_info = []
    
    for article in article_list:
        
        info_dict = {}
        
        response = get(article, headers=headers)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        info_dict['title'] = soup.find('h1').text
        
        info_dict['date_published'] = soup.find('span', class_='published').text

        info_dict['content'] = soup.find('div', class_='entry-content').text
        
        article_info.append(info_dict)
        
    with open(file, 'w') as f:
        
        json.dump(article_info, f)
        
    return article_info



def scrape_one_page(topic):
    
    base_url = 'https://inshorts.com/en/read/'
    
    response = get(base_url + topic)
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    titles = soup.find_all('span', itemprop='headline')
    
    summaries = soup.find_all('div', itemprop='articleBody')
    
    summary_list = []
    
    for i in range(len(titles)):
        
        temp_dict = {}
        
        temp_dict['title'] = titles[i].text
        
        temp_dict['content'] = summaries[i].text
        
        temp_dict['category'] = topic
        
        summary_list.append(temp_dict)
        
    return summary_list

def get_news_articles():
    
    file = 'news_articles.json'
    
    if os.path.exists(file):
        
        with open(file) as f:
            
            return json.load(f)
    
    topic_list = ['business', 'sports', 'technology', 'entertainment']
    
    final_list = []
    
    for topic in topic_list:
        
        final_list.extend(scrape_one_page(topic))
        
    with open(file, 'w') as f:
        
        json.dump(final_list, f)
        
    return final_list    