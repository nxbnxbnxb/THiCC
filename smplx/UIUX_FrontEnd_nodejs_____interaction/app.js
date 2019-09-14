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

// By default, the client will authenticate using the service account file
// specified by the GOOGLE_APPLICATION_CREDENTIALS environment variable and use
// the project specified by the GOOGLE_CLOUD_PROJECT environment variable. See
// https://github.com/GoogleCloudPlatform/google-cloud-node/blob/master/docs/authentication.md
// These environment variables are set automatically on Google App Engine
const {Storage} = require('@google-cloud/storage');












//===============================================================================================================
//===============================================================================================================
//===============================================================================================================
/*
 *==================================================================================================
 *  Start.
 *    Nathan's additions:
 *      https://www.w3schools.com/nodejs/nodejs_uploadfiles.asp
 *
 *
 *===================================================================================================
 */



//*
const http = require('http');
const formidable = require('formidable');
const s = require('fs');
// */


/*
 *===================================================================================================
 *  End.
 *    Nathan's additions:
 *
 *
 *
 *===================================================================================================
 */
//===============================================================================================================
//===============================================================================================================
//===============================================================================================================











//===============================================================================================================
//                Begin "real code"   (imperative statements and funciton declarations)
//===============================================================================================================
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
  res.render('form.pug');
});

//===============================================================================================================
// Process the file upload and upload to Google Cloud Storage.
app.post('/upload', multer.single('file'), (req, res, next) => {
  /*
   *===================================================================================================
   *    BEGIN DOCSTRING:
   *===================================================================================================
   *
   *  Nathan (nxb) doesn't know Node or javascript.  So all these comments to help him are from https://expressjs.com/en/guide/writing-middleware.html.  If there are any technical mistakes, it's all from the author(s) of https://expressjs.com/en/guide/writing-middleware.html.  But I'm going to assume the authors of expressjs (almost completely) know what they're talking about.
   *  "Middleware functions are functions that have access to the request object (req), the response object (res), and the next function in the application’s request-response cycle."
   *
   *
   *
   *  Params:
   *    "req"   means "request object"
   *    "res"   means "result  "
   *    "next"  "Callback argument to the middleware function, called "next" by convention.
   *            "The next function is a function in the Express router which, when invoked, executes the middleware succeeding the current middleware."
   *
   *===================================================================================================
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
  //===============================================================================================================
  /*
   *==================================================================================================
   *  Start.
   *    Nathan's additions:
   *      https://www.w3schools.com/nodejs/nodejs_uploadfiles.asp
   *===================================================================================================
   */

    /*
    var form = new formidable.IncomingForm();
    form.parse(req, function (err, fields, files) {
      res.write('File uploaded');
      //console.log(req.file);
      res.end();   // they wanted this in  https://www.w3schools.com/nodejs/nodejs_uploadfiles.asp.   But I don't want the page to "end()" after I get the file locally
    });
    res.status(200).send(publicUrl);

  blobStream.end(req.file.buffer);        // NOTE:  this didn't have to call "next()" because this counts as a "response?"  Or maybe the "response" referred to in "https://expressjs.com/en/guide/writing-middleware.html" is "res.status(200).send(publicUrl); " 

    */


  /*
   *  What is in req.file?            req == "request" (what the client sends US)
   *      console.log(req.file)   output:
   *
   *   { fieldname: 'file',
   *     originalname: 'pi.txt',
   *     encoding: '7bit',
   *     mimetype: 'text/plain',
   *     buffer: <Buffer 0a 20 20 78 2e 0a>,
   *     size: 6 }
   */






  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.
  // TODO:  We wanna try to avoid making a high-detail large-file-size **image** be sent too many times.





  // TODO: remove these comments.  (get rid of the cloud storage bucket upload)
  //*     Instead of uploading the file, we should call SMPL-X on it.
 //*    But prob we should upload the file FIRST, then call python on it, then have the user wait somehow, THEN return the GUI with the user in the new clothes.


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
    res.status(200).send(publicUrl);
  });

  blobStream.end(req.file.buffer);        // NOTE:  this didn't have to call "next()" because this counts as a "response?"  Or maybe the "response" referred to in "https://expressjs.com/en/guide/writing-middleware.html" is "res.status(200).send(publicUrl); " 
   //*/
});
//===============================================================================================================

const PORT = process.env.PORT || 8081; // 8080;
app.listen(PORT, () => {
  console.log(`App listening on port ${PORT}`);
  console.log('Press Ctrl+C to quit.');
});
// [END gae_storage_app]

module.exports = app;
//===============================================================================================================



















