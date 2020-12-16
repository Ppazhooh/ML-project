#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import general purpose packages
import numpy as np 
import pandas as pd 
import sys
import json
import os
import re

# This is what we are using for data preparation and ML part (thanks, Rafal, for great tutorial)
from sklearn.preprocessing import LabelBinarizer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras.models import Sequential

from keras.layers import Activation, Dense,Dropout


# In[2]:


temp_artists = pd.read_csv('artists-data.csv')
songs = pd.read_csv('lyrics-data.csv') #load the list of songs


# In[3]:


A=[]
for i in range(len(temp_artists['Genres'])):
    temp=str(temp_artists['Genres'].iloc[i])
    temp2=str(temp_artists['Genre'].iloc[i])
    if ('Rock' in temp and 'Pop' in temp and 'Rock' in temp2):
        A.append(i)
    elif ('Rock' in temp and 'Pop' in temp and 'Pop' in temp2):
        A.append(i) 
    elif ('Hip Hop' in temp and 'Pop' in temp and 'Pop' in temp2):
        A.append(i)
    elif ('Hip Hop' in temp and 'Pop' in temp and 'Hip Hop' in temp2):
        A.append(i)
    elif ('Hip Hop' in temp and 'Rock' in temp and 'Hip Hop' in temp2):
        A.append(i)
    elif ('Hip Hop' in temp and 'Rock' in temp and 'Rock' in temp2):
        A.append(i)

artists=temp_artists.drop(temp_artists.index[A])
artists.index = range(len(artists))
artists = pd.read_csv('artists-data.csv')


# In[4]:



print(songs)


# In[5]:


data = pd.DataFrame() 
G=artists.Genre.unique()
for genre in G:
    Genre_artists = artists[artists['Genre']==genre] # filter artists by genre
    Genre_songs = pd.merge(songs, Genre_artists, how='inner', left_on='ALink', right_on='Link') #inner join of pop artists with songs to get only songs by pop artists
    Genre_songs = Genre_songs[['Genre', 'Artist', 'SName', 'Lyric','Idiom']].rename(columns={'SName':'Song'})#leave only columns of interest and rename some of them.
    Genre_songs = Genre_songs.dropna() # Remove incomplete records, cleanse lyrics
    #Genre_songs = Genre_songs[songs['Lyric']!='Instrumental'].head(SONGS_PER_GENRE).applymap(cleanse) #Remove instrumental compositions  and limit the size of final dataset
    data=pd.concat([data, Genre_songs])
    
    
data=data.loc[data['Idiom'] == 'ENGLISH']

data.index = range(len(data))


# In[6]:


#Keep only 3 most frequent genrres
data=data.loc[data['Genre'].isin(['Rock','Pop','Hip Hop'])]
data.index = range(len(data))



# In[7]:


#Convert Categorical genre to binary label
GenreBinarizer = LabelBinarizer().fit(data['Genre'])
Genre_Label = GenreBinarizer.transform(data['Genre'])


Genre_Label_df = pd.DataFrame(Genre_Label, columns =['G_H', 'G_P', 'G_R'])

final_data=pd.concat([data, Genre_Label_df], axis=1)

##### Shuffle data
final_data=final_data.sample(frac=1) 
final_data.index = range(len(final_data))
final_data=final_data.drop(columns=[ 'Artist','Idiom'])


# In[8]:


len(final_data)


# In[9]:


#Creat train,validation and test data
train_data=final_data[:99189]
validation_data=final_data[99189:111588]
test_data=final_data[111588:]


# In[10]:


final_data['Genre'].value_counts()


# In[11]:


documents = []
stemmer = WordNetLemmatizer()
for sen in range(0, len(train_data)):
    temp=str(train_data['Lyric'].values[sen])
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(temp))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    

    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
    
    
train_Lyric=documents


# In[12]:


documents = []
stemmer = WordNetLemmatizer()
for sen in range(0, len(validation_data)):
    temp=str(validation_data['Lyric'].values[sen])
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(temp))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    

    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
    
    
validation_Lyric=documents


# In[13]:


documents = []
stemmer = WordNetLemmatizer()
for sen in range(0, len(test_data)):
    temp=str(test_data['Lyric'].values[sen])
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(temp))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    

    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
    
    
test_Lyric=documents


# In[14]:


y_train=train_data[['G_H','G_P','G_R']].to_numpy()
y_validation=validation_data[['G_H','G_P','G_R']].to_numpy()
y_test=test_data[['G_H','G_P','G_R']].to_numpy()


# In[15]:


tfidfconverter = TfidfVectorizer(max_features=2000,min_df=1, max_df=0.9, stop_words=stopwords.words('english'))
#
X_Lyric_train = tfidfconverter.fit_transform(train_Lyric).toarray()

X_Lyric_validation = tfidfconverter.transform(validation_Lyric).toarray()

X_Lyric_test = tfidfconverter.transform(test_Lyric).toarray()


# In[ ]:





# In[125]:


X_train_dim=X_Lyric_train.shape[1]
model = Sequential()
model.add(Dense(20, input_dim=X_train_dim, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(10, activation='relu'))

model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[126]:


history=model.fit(x=X_Lyric_train, y=y_train, validation_data=(X_Lyric_validation,y_validation),epochs=10, batch_size=20)


# In[122]:


from sklearn.metrics import confusion_matrix
predictions = model.predict(X_Lyric_test)
A=confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))



from sklearn.metrics import f1_score

f1_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='micro')


# In[31]:


import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[124]:


plot_confusion_matrix(cm=A , normalize    = True,target_names = ['Hip Hop', 'Pop', 'Rock'],title= "Confusion Matrix on Test Data For 3 Layer NN using TFIDF as features")




# In[37]:


from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()


#y_train=y_train.argmax(axis=1)

#Train the model using the training sets
gnb.fit(X_Lyric_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_Lyric_test)

B=confusion_matrix(y_test.argmax(axis=1), y_pred)
plot_confusion_matrix(cm=B , normalize    = True,target_names = ['Hip Hop', 'Pop', 'Rock'],title= "Confusion Matrix on Test Data for Naive Bayes using TFIDF as features")
f1_score(y_test.argmax(axis=1), y_pred, average='macro')


# In[36]:


plot_confusion_matrix(cm=B , normalize    = True,target_names = ['Hip Hop', 'Pop', 'Rock'],title= "Confusion Matrix on Test Data for Naive Bayes using TFIDF as features")


# In[ ]:


#y_pred = clf.predict(X_Lyric_test)
#B=confusion_matrix(y_test.argmax(axis=1), y_pred)
#plot_confusion_matrix(cm=B , normalize    = True,target_names = ['Hip Hop', 'Pop', 'Rock'],title= "Confusion Matrix for SVM using TFIDF as features")


# In[ ]:


stemmer = WordNetLemmatizer()
document =" These are. These are days you'll remember"
document = document.split()

document = [stemmer.lemmatize(word) for word in document]
document = ' '.join(document)
    
    


# In[ ]:


print(document)


# In[16]:


from sklearn.linear_model import LogisticRegression
y_train=y_train.argmax(axis=1)
clf = LogisticRegression(random_state=0).fit(X_Lyric_train, y_train)
y_pred=clf.predict(X_Lyric_test)


# In[21]:


from sklearn.metrics import f1_score
f1_score(y_test.argmax(axis=1), y_pred, average='macro')


# In[23]:


accuracy_score(y_test.argmax(axis=1), y_pred)


# In[33]:


from sklearn.metrics import confusion_matrix
A=confusion_matrix(y_test.argmax(axis=1), y_pred)
plot_confusion_matrix(cm=A , normalize    = True,target_names = ['Hip Hop', 'Pop', 'Rock'],title= "Confusion Matrix using Logistic Regression with TF-IDF")




# In[ ]:


plt(y_test.argmax(axis=1))


# In[29]:


import matplotlib.pyplot as plt
songs['Idiom'].value_counts()[0:6].plot(kind='bar');


plt.title('Distribution of most frequent languages in data')
plt.ylabel('Number of lyrics in data set')

plt.figure(dpi=1200)


# In[30]:


import matplotlib.pyplot as plt
data['Genre'].value_counts().plot(kind='bar');


plt.title('Distribution of 3 major gernes in final data')
plt.ylabel('Number of lyrics in data set')
plt.figure(dpi=1200)


# In[ ]:




