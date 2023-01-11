from flask import Flask, render_template, request

app = Flask(__name__)
app.debug = True

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        result = request.form

        return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)