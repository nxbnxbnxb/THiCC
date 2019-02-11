# TODO  (copy from /https://www.youtube.com/watch?v=iwxzilyxTbQ&index=3&list=PLXmMXHVSvS-DvYrjHcZOg7262I9sGBLFR or similar videos)
from celery import Celery

def make_celery(app):
    celery = Celery(
        app.import_name,
        #backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


