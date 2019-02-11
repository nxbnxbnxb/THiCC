
from flask import Flask

UPLOAD_FOLDER = '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/customer_imgs'  # changing this didn't fix the bug (Sun Feb 10 12:32:26 EST 2019)
    # NOTE TODO
    # whenever `git pull`ing, change port 5000 to port 80
    #   and "/home/n/" to "/home/n/"
    # and vice versa.


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

