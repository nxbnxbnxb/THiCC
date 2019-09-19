var path = require('path');
var express = require('express'); // version :   express@4.17.1
var app = express();

var dir = path.join(__dirname, 'public');

app.use(express.static(dir));

var port=8082;
app.listen(port, function () {
    console.log('Listening on http://localhost:'+port+'/'));
});









































