import os
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
IMG_FNAME=''

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/upload', methods=['GET', 'POST', 'PUT'])
def upload_form():
	return render_template('upload.html')

@app.route('/uploaded', methods=['GET', 'POST', 'PUT'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File(s) successfully uploaded')
			return redirect('/upload')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)
    # NOTE TODO
    # whenever `git pull`ing, change port 5000 to port 80
    #   and "/home/n/" to "/home/ubuntu/"
    # and vice versa.
