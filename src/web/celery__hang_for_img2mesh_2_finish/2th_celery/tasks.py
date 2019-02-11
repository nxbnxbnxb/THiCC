from celery import Celery
import time

app = Celery("tasks", broker='amqp://localhost//')

@app.task
def reverse(string):
  time.sleep(10)
  return string[::-1]


# This simple example works.  Now the question is: can we extend this to the whole "backend" thing and also serve an .obj file?
