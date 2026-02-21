const http = require('http');

const options = {
    hostname: '127.0.0.1',
    port: 5173,
    path: '/',
    method: 'GET',
    timeout: 5000
};

console.log('Attempting to connect to http://127.0.0.1:5173...');

const req = http.request(options, (res) => {
    console.log(`STATUS: ${res.statusCode}`);
    process.exit(0);
});

req.on('error', (e) => {
    console.error(`PROBLEM WITH REQUEST: ${e.message}`);
    process.exit(1);
});

req.end();
