
from flask import Flask

UPLOAD_FOLDER = '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/simple_upload_flask_3'


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

