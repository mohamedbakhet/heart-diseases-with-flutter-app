from flask import *
from flask import Flask,request, jsonify,render_template,Blueprint

from deep import model


app = Flask(__name__)
m=model()

@app.route('/')
def upload():
    mod = Blueprint('backend', __name__, template_folder='templates', static_folder='./static')
    UPLOAD_URL = 'http: // 192.168.1.103: 5000 / static /'

    return render_template("file_upload_form.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        user_file = request.files.getlist('file')
        for file in user_file:
            path = os.path.join(os.getcwd() + '\\static\\' + file.filename)
            file.save(path)


        # f = request.files['file']
        # f.save(f.filename)
        # f = request.files['file']
        # f.save(f.filename)
        # f = request.files['file']
        # f.save(f.filename)
        x=os.path.join(os.getcwd() + '\\static\\' + file.filename[:-4])

        n = m.predict(x)
        if n == 'Myocardial infraction':
            s=m.predictmyco(x)
            return jsonify({
                "status": "success",
                "prediction": n,
                "confidence": s,
            })
        else:
            return jsonify({
                "status": "success",
                "prediction": n,
                "confidence": None,
            })

        # return render_template("success.html",hame=s,name=n)

#
# def predict():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             return "someting went wrong 1"
#
#         user_file = request.files['file']
#         temp = request.files['file']
#         if user_file.filename == '':
#             return "file name not found ..."
#
#         else:
#             path = os.path.join(os.getcwd() + '\\modules\\static\\' + user_file.filename)
#             user_file.save(path)
#             classes = identifyImage(path)
#             db.addNewImage(
#                 user_file.filename,
#                 classes[0][0][1],
#                 str(classes[0][0][2]),
#                 datetime.now(),
#                 UPLOAD_URL + user_file.filename)
#
#             return jsonify({
#                 "status": "success",
#                 "prediction": classes[0][0][1],
#                 "confidence": str(classes[0][0][2]),
#                 "upload_time": datetime.now()
#             })

if __name__ == '__main__':
    app.run(debug=True)
