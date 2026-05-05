from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Homepage"

@app.route('/niggers')
def niggers():
    return "Welcome nigga"


if __name__ == '__main__':
    app.run(debug=True)