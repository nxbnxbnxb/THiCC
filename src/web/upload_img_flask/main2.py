
import os
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
IMG_FNAME=''

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_mesh(img_fname):
  hmr_demo_loc='/home/n/Documents/code/old/hmr/'
  os.system("conda activate cat && python2 -m {0}demo.py --img_path {1}".format(hmr_demo_loc,img_fname))
  with open(hmr_demo_loc+"mesh.obj", 'r') as fp:
    return fp.read()  # TODO: figure out how obj file will be communicated; sftp?  Pierlorenzo will know

@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
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
      IMG_FNAME= os.path.join(app.config['UPLOAD_FOLDER'], filename) # global
			flash('File(s) successfully uploaded')
			return redirect('/')

if __name__ == "__main__":
    app.run()
    make_mesh(IMG_FNAME)
