from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      state = request.form['state']
      result1 = request.form['vehicle1']
      return render_template("result.html",result = state, result1=result1)

if __name__ == '__main__':
   app.run(debug = True)