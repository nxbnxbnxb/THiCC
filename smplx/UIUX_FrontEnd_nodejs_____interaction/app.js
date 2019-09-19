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

const http = require('http');
const formidable = require('formidable');
const s = require('fs');

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

//===============================================================================================================
// Display a form for uploading files.
app.get('/', (req, res) => {
  res.render('show.pug');
});


const PORT = process.env.PORT || 8081; // 8080;
app.listen(PORT, () => {
  console.log(`App listening on port ${PORT}`);
  console.log('Press Ctrl+C to quit.');
});
// [END gae_storage_app]

module.exports = app;
//===============================================================================================================












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







//*







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
