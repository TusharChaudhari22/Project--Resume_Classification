# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 18:12:20 2023

@author: Tushar
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import docx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud, STOPWORDS

import warnings
warnings.filterwarnings('ignore')

# Title
st.title("Resume Classification :bookmark_tabs:")

# Store file.
path = r'D:\Project\Uploaded_Resumes'

# Upload file.
uploaded_file = st.file_uploader("Choose a file", type=['Docx'])
if uploaded_file is not None:
  
    
    # Extract file path
    name = uploaded_file.name
    file_path = f"{path}\{name}"
    
    
    # Store file in local disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    
    # Read text data from file.
    resume = []
    doc = docx.Document(file_path)
    for para in doc.paragraphs:
        resume.append(para.text)
    resume = ' '.join(resume)
    st.subheader('Preview')
    st.write(resume)
    
    
    # Import data to train the model.
    Data = pd.read_csv('Resumes.csv')
    
    
    # Created dataframe for uploaded file
    data = {'Resumes' : resume,
            'Category' : None}
    df = pd.DataFrame(data, index=[0])
    
    
    # Merge both dataframe
    Data = pd.concat([Data,df], axis=0)
    Data.reset_index(drop=True, inplace=True)
    Data = Data.drop(columns='Unnamed: 0',axis=1)
    
#==============================================================================   
    
    # Text preprocessing steps
    
    # 1) Remove special characters, numbers and punctuations.
    Data['Cleaned_Resume'] = Data['Resumes'].str.replace("[^a-zA-Z]", " ")
    
    # 2) Remove links.
    Data['Cleaned_Resume'] = Data['Cleaned_Resume'].str.replace('http[^\s][^s]+', " ")
    
    # 3) Text Normalization.
    Data['Cleaned_Resume'] = Data['Cleaned_Resume'].apply(lambda x: ' '.join(i.lower() for i in x.split()))
    
    # 4) Remove English stopwords.
    Data['Cleaned_Resume'] = Data['Cleaned_Resume'].apply(lambda x: ' '.join([i for i in x.split() if i not in stopwords.words('english')]))
    
    # 5) Lemmatization.
    lemma = WordNetLemmatizer()
    Data['Cleaned_Resume'] = Data['Cleaned_Resume'].apply(lambda x: ' '.join(lemma.lemmatize(i) for i in x.split()))
    
    # 6) Remove least frequent words.
    freq = pd.Series(' '.join(Data['Cleaned_Resume']).split()).value_counts()
    least_freq = freq[freq.values == 1].index
    Data['Cleaned_Resume'] = Data['Cleaned_Resume'].apply(lambda x: ' '.join(i for i in x.split() if i not in least_freq))
    
#==============================================================================   
    
    # Skills
    st.subheader('Skills')
    text = ' '.join(Data['Cleaned_Resume'][77:])
    
    
    # Defined required skills.
    skills = ['html','css','jsx','react','javascript','git','node','npm','redux','rdbms','json','python','rest','graphql','swagger','gcp',
               'mysql','sql','mssql','ssis','ssrs','ssas''rtl','oracle','spark','mongodb','apache','agile','jquery','scrum','plsql','database','tsql',
               'hcm','crm','anp','sqr','peoplecode','hr','payroll','hrms','sdlc',
               'sap','birt','forecasting',
               'word','excel','powerpoint',
               'communication','problemsolving','analytical','troubleshooting', 'debugging']
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    word_list = text.split(' ')
    
    
    # Match required skills with student skills.
    Skills_list = []
    for i in word_list:
        if i in skills:
            Skills_list.append(i)
    
    
    #Count plot for skills present in resume.
    ax = sns.countplot(Skills_list)
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=90)
    st.pyplot()
    
#==============================================================================
    
    # Wordcloud
    
    st.set_option('deprecation.showPyplotGlobalUse', False) # Remove warnings
    
    st.subheader('Wordcloud')
    
    #Frequency of all words present in resume.
    wordcloud = WordCloud(width=3000, height=2000, background_color='black', stopwords=STOPWORDS).generate(text)
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off');
    st.pyplot()
    
#=============================================================================    

    # Feature Extraction using TFIDF Vectorizer.
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(Data['Cleaned_Resume'])
    x.toarray()
    tfidf.get_feature_names_out()
    
    df2 = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names_out())
    df2['Category'] = Data['Category']
    
#==============================================================================
    
    # Split data into train and predict.
    
    # Training data
    train = df2.iloc[:-1,:]
    le = LabelEncoder()

    train['Category'] = le.fit_transform(train['Category'])
    
    # Data for prediction.
    predict_data = df2.iloc[77:,:-1]
    
    # Model building using SVC 
    xtrain = train.iloc[:,:-1]
    ytrain = train['Category']
    svc = SVC()

    svc.fit(xtrain,ytrain)
    
    #Save model
    pickle.dump(svc, open('model.pkl', 'wb'))
    
    #Load model
    load = open('model.pkl','rb')
    model = pickle.load(load)
    
#==============================================================================   
    
    # Predict the category for uploaded resume file.
    st.subheader('Check Resume Category')
    if st.button('Press'):
        result = model.predict(predict_data)
        if result == 0:
            result = 'PeopleSoft'
        elif result == 1:
            result = 'React JS Developer'
        elif result == 2:
            result = 'SQL Developer'
        else:
            result = 'Workday'
        st.success('Category : {} '.format(result))

#==============================================================================   
