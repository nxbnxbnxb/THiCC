# From tutorial at https://www.youtube.com/watch?v=lOirTBrOek0&list=PLXmMXHVSvS-DvYrjHcZOg7262I9sGBLFR&index=4
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from flask_celery import make_celery

import os
import time
import matplotlib.pyplot as plt

import demo # hmr by Akanazawa, Black, et al.

#=====================================================================
# used to be in separate file "app.py"
UPLOAD_FOLDER = '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'obj'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
all_dirs=demo.outmesh_path.split('/')
app.config['MESH_DIR'  ]=demo.outmesh_path[:mesh_path.rfind('/')+1 ]
app.config['MESH_FNAME']=demo.outmesh_path[ mesh_path.rfind('/')+1:]
# end what used to be in separate file "app.py"
#=====================================================================

app.config['CELERY_BROKER_URL']= 'amqp://localhost//'
#app.config['CELERY_RESULT_BACKEND']= 'amqp://localhost//'  # TODO: change this if it's not working.  Hopefully backend isn't 100% necessary, though
celery = make_celery(app)

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
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      return redirect(url_for('uploaded_file',
                              filename=filename))
  return '''
  <!doctype html>
  <title>Upload new File</title>
  <h1>Upload new File</h1>
  <form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <input type=submit value=Upload>
  </form>
  '''

@app.route('/uploads/<filename>')
def uploaded_file(customer_img_fname):
  flash('Customer image file uploaded.  Building realistic 3-D mesh of the customer...')
  demo.make_mesh(customer_img_fname)
  return send_from_directory(app.config['MESH_DIR'],app.config['MESH_FNAME'])
  #return 'Image successfully uploaded'
  #return send_from_directory(app.config['UPLOAD_FOLDER'],   filename) # this returns the previously uploaded file (image)

@celery.task(name='celery_example.make_mesh')
def make_mesh(customer_img_fname):
  os.system("./make_mesh.sh {0}".format(customer_img_fname))# /home/n/Documents/code/old/hmr/demo.py")
  #print("type(customer_img): \n",type(customer_img))  #this worked
  return 'mesh' #TODO: return the .obj file?  Then we'd have to hang with the process

if __name__=="__main__":
  app.run( host='0.0.0.0', debug=True, port=5000)







































































