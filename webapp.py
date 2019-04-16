from flask import Flask, request, render_template, jsonify
import os
import json
from classify import ClassifyImage, loadClassifier


app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = './static/images'

##########
# Simple webapp for Deep Learning Project
##########

c = loadClassifier(_saved_dict='./69_resnet34.pt')
# c = loadClassifier()

# landing page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# upload link
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']

    # path here
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(f)
    label, _ = ClassifyImage(c, _image=f)

    prediction = label
    return render_template('prediction.html',
                           uploaded_image=file.filename,
                           prediction_class=prediction)


# landing page
@app.route('/precomputed', methods=['GET'])
def precomputed():

    with open('data69.json', 'r') as f:
        data = json.loads(f.read())

    return render_template('precomputed.html',
                           val_data=json.dumps(data))



if __name__=='__main__':
    app.run(port=8080)
