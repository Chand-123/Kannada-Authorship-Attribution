# coding=utf-8
import random
from flask import Flask, redirect, url_for, request
# import lib as m

from firebase import firebase
firebase = firebase.FirebaseApplication('https://hackathon-ab821.firebaseio.com/', None)

app = Flask(__name__)

import html

@app.route('/success/<name>')
def success(name):
    texts = str(name)
    k2 = texts.replace("\n", "").replace("\r", "")

    k2 = html.unescape(k2)
    file = open("data/comment.txt", "w")
    file.writelines(k2)
    file.writelines('\n')
    file.close()
    file = open("data/comment.txt", "r")
    txt = file.read()
    
    file.close()
    if type(txt) == str:
        t = ['a']
        t[0]= txt
    else:
        t = txt

    res = m.check_sentiment(t)
    
    k=0
    if res[k] == 0:
        res[k]= 'happy'
    elif res[k] == 1:
        res[k]= 'sad'
    elif res[k] == 2:
        res[k]= 'angry'
    elif res[k] == 3:
        res[k]= 'fear'
    elif res[k] == 4:
        res[k]= 'surprise'
    elif res[k] == 5:
        res[k]= 'neutral'

    # print(res)
    # print(type(txt))

    return 'Comment Entered By User:    '+name+ '<br>Sentiment is <strong>'+ str(res[0]).upper()+'<br>Probability : <br>[Happy,Sad,Angry,Fear,Surprise] => '+str(res[1:])


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name=user))

    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))


if __name__ == '__main__':
    app.run(debug=True)
