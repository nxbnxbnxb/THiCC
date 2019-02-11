#!/bin/bash

celery -A celery_example.celery worker --loglevel=info

