# From tutorial at https://www.youtube.com/watch?v=lOirTBrOek0&list=PLXmMXHVSvS-DvYrjHcZOg7262I9sGBLFR&index=4
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import os
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/ubuntu/Documents/code/old/hmr") # sudo doesn't have the same environment variables from regular bash
import demo # hmr by Akanazawa, Black, et al.

#=====================================================================
# used to be in separate file "app.py"
IMG_UPLOAD_DIR = '/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'obj'])

app = Flask(__name__)
app.config['IMG_UPLOAD_DIR'] = IMG_UPLOAD_DIR
all_dirs=demo.outmesh_path.split('/')
app.config['MESH_DIR'  ]=demo.outmesh_path[:demo.outmesh_path.rfind('/')+1 ]
app.config['MESH_FNAME']=demo.outmesh_path[ demo.outmesh_path.rfind('/')+1:]
# end what used to be in separate file "app.py"
#=====================================================================


def pltshow(x):
  plt.imshow(x); plt.show(); plt.close()
def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['IMG_UPLOAD_DIR'], filename))
      return redirect(url_for('uploaded_file',
                              customer_img_fname=filename))
  return '''
  <!doctype html>
  <title>Upload new File</title>
  <h1>Upload new File</h1>
  <form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <input type=submit value=Upload>
  </form>
  '''

@app.route('/uploads/<customer_img_fname>')
def uploaded_file(customer_img_fname):
  #flash('Customer image file uploaded.  Building realistic 3-D mesh of the customer...')
  demo.make_mesh(app.config['IMG_UPLOAD_DIR']+customer_img_fname)
  print("mesh name is {0}{1}".format(app.config['MESH_DIR'],app.config['MESH_FNAME']))
  return redirect('/mesh/{0}'.format(app.config['MESH_FNAME']))
@app.route('/mesh/<body_mesh_fname>')
def mesh(body_mesh_fname):
  return send_from_directory(app.config['MESH_DIR'],app.config['MESH_FNAME'])

if __name__=="__main__":
  app.run( host='0.0.0.0', debug=True, port=80)







































































