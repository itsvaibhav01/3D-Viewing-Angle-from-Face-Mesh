import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import json
import cv2
import io
import base64
import re
from PIL import Image
import numpy as np

## project files
from main import Proctor

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def home():
    if request.method == 'GET':
        return render_template('index.html')


watcher = Proctor(frame_width=640, frame_height=480, channels=3)

@app.route('/api', methods=['GET', 'POST'])
def api():
    if request.method == "GET":
        return "API GET"

    elif request.method == "POST":
        j = json.loads(request.data)
        webcam = j['webcam_img']
        webcam = re.sub('data:image/jpeg;base64,' , '', webcam)
        webdata = base64.b64decode(webcam)

        image = Image.open(io.BytesIO(webdata))

        image_np = np.array(image)

        obj = watcher.get_stats(img=image_np)

        return jsonify(obj)

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5001, debug=True, ssl_context=('cert.pem', 'key.pem'))