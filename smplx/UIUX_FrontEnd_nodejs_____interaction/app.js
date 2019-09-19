/**
 * Copyright 2016, Google, Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//===============================================================================================================

'use strict';

/*
 *===============================================================================================================
 *  Imports:
 *===============================================================================================================
 */
const process = require('process'); // Required to mock environment variables

// [START gae_storage_app]
const {format} = require('util');
const express = require('express');
const Multer = require('multer');
const bodyParser = require('body-parser');
const nodemailer = require('nodemailer');


// By default, the client will authenticate using the service account file
// specified by the GOOGLE_APPLICATION_CREDENTIALS environment variable and use
// the project specified by the GOOGLE_CLOUD_PROJECT environment variable. See
// https://github.com/GoogleCloudPlatform/google-cloud-node/blob/master/docs/authentication.md
// These environment variables are set automatically on Google App Engine
const {Storage} = require('@google-cloud/storage');

const http = require('http');
const formidable = require('formidable');
const fs = require('fs');
// Instantiate a storage client
const storage = new Storage();

const app = express();
app.set('view engine', 'pug');
app.use(bodyParser.json());

//===============================================================================================================
// Multer is required to process file uploads and make them available via
// req.files.
const multer = Multer({
  storage: Multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024, // no larger than 5mb, you can change as needed.
  },
});

//===============================================================================================================
// A bucket is a container for objects (files).
const bucket = storage.bucket(process.env.GCLOUD_STORAGE_BUCKET);

//===============================================================================================================
// Display a form for uploading files.
app.get('/', (req, res) => {
  res.render('img_upload.pug');
});


//===============================================================================================================
// Process the file upload and upload to Google Cloud Storage.
app.post('/upload', multer.single('file'), (req, res, next) => {
  /*
   *===================================================================================================
   *    BEGIN DOCSTRING:
   *
   *  Nathan (nxb) doesn't know Node or javascript.  So all these comments to help him are from https://expressjs.com/en/guide/writing-middleware.html.  If there are any technical mistakes, it's all from the author(s) of https://expressjs.com/en/guide/writing-middleware.html.  But I'm going to assume the authors of expressjs (almost completely) know what they're talking about.
   *  "Middleware functions are functions that have access to the request object (req), the response object (res), and the next function in the application’s request-response cycle."
   *
   *  Params:
   *    "req"   means "request object"
   *    "res"   means "result  "
   *    "next"  "Callback argument to the middleware function, called "next" by convention.
   *            "The next function is a function in the Express router which, when invoked, executes the middleware succeeding the current middleware."
   *
   *   END DOCSTRING:
   *===================================================================================================
   *
   */
  //===============================================================================================================
  // Error checking (if invalid file)
  if (!req.file) {
    res.status(400).send('No file uploaded.');
    return;
  }
  // Create a new blob in the bucket and upload the file data.
  const blob = bucket.file(req.file.originalname);
  const blobStream = blob.createWriteStream({
    resumable: false,
  });
  blobStream.on('error', err => {
    next(err);        // "next()"   calls the next "middleware function" (https://expressjs.com/en/guide/writing-middleware.html)
  });

  blobStream.on('finish', () => {
    // The public URL can be used to directly access the file via HTTP.
    const publicUrl = format(
      `https://storage.googleapis.com/${bucket.name}/${blob.name}`
    );
    res.render('email.pug');
    // TODO:   change this message.
  });
  blobStream.end(req.file.buffer);
});
//===============================================================================================================

//===============================================================================================================
app.get('/email_upload', multer.single('file'), (req, res, next) => {
  var customer_email = req.query.email_field; //mytext is the name of your input box            
  //console.log(typeof customer_email); // string

  fs.writeFile('cust_email.txt', customer_email, (err) => {
    // throws an error, you could also catch it here
    if (err) throw err;
    // success case, the file was saved:
    else console.log('Email address saved!');
  });
  var msg_2_customer="<html><body><h1>Please check your e-mail: "+customer_email+"</h1>  <h4>in 1-2 minutes, you can shop for clothes in 3-D!</h4></body></html>";
  res.status(200).send(msg_2_customer);
});
//===============================================================================================================
 


const PORT = process.env.PORT || 8081; // 8080;
app.listen(PORT, () => {
  console.log(`App listening on port ${PORT}`);
  console.log('Press Ctrl+C to quit.');
});
// [END gae_storage_app]

module.exports = app;
