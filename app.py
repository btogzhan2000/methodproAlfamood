from flask import Flask,render_template,url_for,request, jsonify
from flask_bootstrap import Bootstrap 
import pandas as pd
import pymysql.cursors
import atexit
from apscheduler.scheduler import Scheduler
import datetime
import time

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pymystem3 import Mystem
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
Bootstrap(app)


# Explicitly kick off the background thread

conn = pymysql.connect(host='sql12.freemysqlhosting.net',
                             user='sql12301018',
                             password='E9DrEiTwRC',
                             db='sql12301018',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
 
cursor = conn.cursor()
sql_timezone = """ set time_zone='+6:00';"""
cursor.execute(sql_timezone) 
cron = Scheduler(daemon=True)
# Explicitly kick off the background thread
cron.start()
model = gensim.models.KeyedVectors.load_word2vec_format('models/model.bin', binary=True)
mystem = Mystem() 
def remov_punct(withpunct):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        without_punct = ""
        char = 'nan'
        for char in withpunct:
                if char not in punctuations:
                        without_punct = without_punct + char
        return (without_punct)

stop_rus = set(stopwords.words('russian'))
df = pd.DataFrame()


@cron.interval_schedule(minutes=30)
def job_function():
        global  df 
        df = pd.read_sql_query("SELECT mood_rate, mood_comment, department_name FROM mood JOIN department ON mood.department_id = department.department_id", conn)
        
        list = []
        for s in range(len(df)):
                list.append(remov_punct(df['mood_comment'][s]).lower())

        word_tokens = []
        for line in list:
                word_tokens.append(word_tokenize(line))

        without_stop = []
        for line in word_tokens:
                temp_line = []
                for word in line:
                        if word not in stop_rus:
                                temp_line.append(word)
                without_stop.append(temp_line)

        test=[]
        test2=[]
        for sentence in without_stop:
                for word in sentence:
                        test.append(mystem.lemmatize(word))
                test2.append(test)
                test=[]

        lst1=[]
        lst2=[]
        for part in test2:
                for sentence in part:
                        for pair in sentence:
                                if pair is not '\n':
                                        lst1.append(pair)
                lst2.append(lst1)
                lst1=[],
        text=[]
        for sentence in lst2:
                for word in sentence:
                        test.append(word)
                text.append(test)
                test=[]

        for sentence in lst2:
                test.append(TreebankWordDetokenizer().detokenize(sentence))

        df['text'] = test
        print("executed")
        


atexit.register(lambda: cron.shutdown(wait=False))

job_function()
def function():
    cursor = conn.cursor() 
    sql_happy = """ SELECT COUNT(*) FROM mood WHERE mood_rate='happy';"""
    sql_good = """ SELECT COUNT(*) FROM mood WHERE mood_rate='good';"""
    sql_meh = """ SELECT COUNT(*) FROM mood WHERE mood_rate='meh';"""
    sql_sad = """ SELECT COUNT(*) FROM mood WHERE mood_rate='sad';"""
    sql_angry = """ SELECT COUNT(*) FROM mood WHERE mood_rate='angry';""" 
    sql_total = """ SELECT COUNT(*) FROM mood;"""

    cursor.execute(sql_total)
    total = cursor.fetchall()
    global total_number 
    total_number = total[0]['COUNT(*)']


    cursor.execute(sql_happy) 
    global result_happy 
    result_happy = cursor.fetchall()
    global percent_happy 
    percent_happy = str(round(result_happy[0]['COUNT(*)'] / total_number * 100, 2))

    cursor.execute(sql_good) 
    global result_good
    result_good = cursor.fetchall()
    global percent_good
    percent_good = str(round(result_good[0]['COUNT(*)'] / total_number * 100, 2))

    cursor.execute(sql_meh) 
    global result_meh 
    result_meh = cursor.fetchall() 
    global percent_meh
    percent_meh = str(round(result_meh[0]['COUNT(*)'] / total_number * 100, 2))

    cursor.execute(sql_sad) 
    global result_sad
    result_sad = cursor.fetchall() 
    global percent_sad
    percent_sad = str(round(result_sad[0]['COUNT(*)'] / total_number * 100, 2))

    cursor.execute(sql_angry) 
    global result_angry
    result_angry = cursor.fetchall() 
    global percent_angry
    percent_angry = str(round(result_angry[0]['COUNT(*)'] / total_number * 100, 2))

    print('done')

function()



@app.route('/index')
def index():
        
    
    #return jsonify(n_rows_sad = result_sad[0]['COUNT(*)'])
        return render_template('index.html', n_rows_happy=result_happy[0]['COUNT(*)'], n_rows_good=result_good[0]['COUNT(*)'],
                          n_rows_meh=result_meh[0]['COUNT(*)'], n_rows_sad=result_sad[0]['COUNT(*)'],
                          n_rows_angry=result_angry[0]['COUNT(*)'], percent_happy = percent_happy,
                          percent_good = percent_good, percent_meh = percent_meh, percent_sad = percent_sad,
                          percent_angry = percent_angry, total_number = total_number)

                          
@app.route('/stats')
def stats():
        
        return render_template('wordStats.html')

@app.route('/login')
def login():
        return render_template('login.html')


@app.route('/moodstats', methods=['POST', 'GET'])
def moodstats():
        if request.method == 'POST':
                date = request.form['datee']
                print(date)
                date_splitted = date.split(' - ')
                start_date = date_splitted[0]
                end_date = date_splitted[1]
                
                start = start_date.split('/')
                month = start[0]
                day = start[1]
                year = start[2]
                start_date_sql = year + '-' + month + '-' + day
                print(start_date_sql)

                end = end_date.split('/')
                month = end[0]
                day = end[1]
                year = end[2]
                end_date_sql = year + '-' + month + '-' + day
                print(end_date_sql)

                date_sql = '''SELECT COUNT(*) FROM mood
                WHERE DATE(mood_created) BETWEEN (%s) AND (%s);'''
                date_sql2 = '''SELECT * FROM mood
                WHERE DATE(mood_created) >= (%s) AND DATE(mood_created) <= (%s);'''
                

                date_happy_sql = '''SELECT COUNT(*) FROM mood
                WHERE DATE(mood_created) BETWEEN (%s) AND (%s) AND mood_rate='happy';'''
                date_good_sql = '''SELECT COUNT(*) FROM mood
                WHERE DATE(mood_created) BETWEEN (%s) AND (%s) AND mood_rate='good';'''
                date_meh_sql = '''SELECT COUNT(*) FROM mood
                WHERE DATE(mood_created) BETWEEN (%s) AND (%s) AND mood_rate='meh';'''
                date_sad_sql = '''SELECT COUNT(*) FROM mood
                WHERE DATE(mood_created) BETWEEN (%s) AND (%s) AND mood_rate='sad';'''
                date_angry_sql = '''SELECT COUNT(*) FROM mood
                WHERE DATE(mood_created) BETWEEN (%s) AND (%s) AND mood_rate='angry';'''
                date_total_sql = '''SELECT COUNT(*) FROM mood
                WHERE DATE(mood_created) BETWEEN (%s) AND (%s);'''

                cursor = conn.cursor() 

                cursor.execute(date_happy_sql, (start_date_sql, end_date_sql, ))
                date_happy = cursor.fetchall()

                cursor.execute(date_good_sql, (start_date_sql, end_date_sql, ))
                date_good = cursor.fetchall()

                cursor.execute(date_meh_sql, (start_date_sql, end_date_sql, ))
                date_meh = cursor.fetchall()

                cursor.execute(date_sad_sql, (start_date_sql, end_date_sql, ))
                date_sad = cursor.fetchall()

                cursor.execute(date_angry_sql, (start_date_sql, end_date_sql, ))
                date_angry = cursor.fetchall()

                cursor.execute(date_total_sql, (start_date_sql, end_date_sql, ))
                date_total = cursor.fetchall()
                date_total_number = date_total[0]['COUNT(*)']
                print(date_total_number)
                
                if date_total_number is not 0:
                        date_percent_happy = str(round(date_happy[0]['COUNT(*)'] / date_total_number * 100, 2))
                        date_percent_good = str(round(date_good[0]['COUNT(*)'] / date_total_number * 100, 2))
                        date_percent_meh = str(round(date_meh[0]['COUNT(*)'] / date_total_number * 100, 2))
                        date_percent_sad = str(round(date_sad[0]['COUNT(*)'] / date_total_number * 100, 2))
                        date_percent_angry = str(round(date_angry[0]['COUNT(*)'] / date_total_number * 100, 2))
                else:
                        date_percent_happy = str(0)
                        date_percent_good = str(0)
                        date_percent_meh = str(0)
                        date_percent_sad = str(0)
                        date_percent_angry = str(0)
                     

        return render_template('index.html', n_rows_happy=result_happy[0]['COUNT(*)'], n_rows_good=result_good[0]['COUNT(*)'],
                n_rows_meh=result_meh[0]['COUNT(*)'], n_rows_sad=result_sad[0]['COUNT(*)'],
                n_rows_angry=result_angry[0]['COUNT(*)'], percent_happy = percent_happy,
                percent_good = percent_good, percent_meh = percent_meh, percent_sad = percent_sad,
                percent_angry = percent_angry, total_number = total_number,
                date_happy=str(date_happy[0]['COUNT(*)']), date_good=str(date_good[0]['COUNT(*)']),
                date_meh=str(date_meh[0]['COUNT(*)']), date_sad=str(date_sad[0]['COUNT(*)']),
                date_angry=str(date_angry[0]['COUNT(*)']), date_percent_happy=date_percent_happy,
                date_percent_good=date_percent_good, date_percent_meh=date_percent_meh,
                date_percent_sad=date_percent_sad, date_percent_angry=date_percent_angry)

@app.route('/predict', methods=['POST'])
def predict():
            
        if request.method == 'POST':
                department2 = request.form['department2']    
                mood2 = request.form['mood2']  
        data_sorted = pd.DataFrame()
        data_sorted = df[(df.mood_rate==mood2) & (df.department_name==department2)]

        '''
        data_sorted = pd.DataFrame()
        data_sorted = pd.read_sql_query("SELECT mood_rate, text, department_name FROM mood JOIN department ON mood.department_id = department.department_id WHERE mood.mood_rate= (%s) AND department.department_name=(%s)", conn, params=(mood2, department2))
        print(data_sorted)
        '''


        list=[]
        list = data_sorted.text.tolist()

        word_tokens2 = []
        for line in list:
                word_tokens2.append(word_tokenize(line))

        list=[]
        for sentence in word_tokens2:
                for word in sentence:
                        list.append(word)


        def return_matrix(random_words, dim=300):
                word_matrix = np.random.randn(len(random_words), dim)
                i = 0
                for word in random_words:
                        word_matrix[i] = model[word]
                        i += 1
                return word_matrix

        def find_word(word):
                for key in model.vocab.keys():
                        x = re.search(word, key)
                        if (x):
                                return key

        random_words_test = set(list)
        random_words = []
        for word in random_words_test:
                temp_word = find_word(word)
                if temp_word is not None:
                        random_words.append(temp_word)

        words = []
        for word in random_words:
                x = re.search("NOUN$", word)
                if (x):
                        words.append(word.split("_")[0])

        words2 = []
        for word in random_words:
                x = re.search("NOUN$", word)
                if (x):
                        words2.append(word)
        return_matrix_ = return_matrix(words2)

        tsne = TSNE(n_components=2, verbose=1, perplexity=2, method='exact')
        tsne_results = tsne.fit_transform(return_matrix_)
        
        plt.figure(figsize=(16, 16))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], cmap=plt.get_cmap('Spectral'))

        for label, x, y in zip(words, tsne_results[:, 0], tsne_results[:, 1]):
                plt.annotate(
                label,
                xy=(x, y),
                xytext=(-14, 14),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )

        plt.xlabel('TSNE Component 1 ')
        plt.ylabel('TSNE Component 2')
        plt.title('TSNE график для репрезентации анализа комментариев')
        graph_name = 'hi_' + str(time.time()) + '.png'
        
        
        plt.savefig('static/' + graph_name)
        return render_template('result.html', department = department2, mood = mood2, graph=graph_name)

if __name__ == '__main__':
    app.run(debug=True)
