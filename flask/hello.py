from flask import Flask
app = Flask(__name__)


# 경로 및 반환 데이터 타입
@app.route('/') 
def index():
    return 'Index Page'

@app.route('/hello') 
def hello():
    return 'Hello World'

@app.route('/return_number') 
def return_number():
    data = 100
    return 'data=%d'%data


# html 파일 연결하기
from flask import render_template
@app.route('/test')
def test():
    return render_template("test.html")


# get과 post 방식으로 받기
from flask import request, redirect, url_for
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'error'
        else:
            error = 'success'
    return render_template('login.html', error=error)


# ajax 방식
@app.route('/ajax', methods=['GET', 'POST'])
def ajax():
    if request.method == "POST":
        jsonData = request.get_json()
        name = jsonData["name"]
        return name+" Hello"
    return render_template("ajax.html")




if __name__ == '__main__':
    app.run()




