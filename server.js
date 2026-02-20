const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PORT = process.env.PORT || 3456;
const MEAN = [0.48145466, 0.4578275, 0.40821073];
const STD = [0.26862954, 0.26130258, 0.27577711];
const ALLOWED = ['https://www.nigooffice.com','https://nigooffice.com','http://localhost:3456'];

// Large files hosted externally - downloaded on first startup
const FILES = {
  'model.onnx': process.env.MODEL_URL || '',
  'vs-embeddings.bin': process.env.EMBEDDINGS_URL || '',
  'vs-meta.json': process.env.META_URL || '',
};

let session, meta, embeddings;

async function downloadIfMissing(filename, url) {
  const fp = path.join(__dirname, filename);
  if (fs.existsSync(fp)) return console.log(`  ${filename}: exists`);
  if (!url) throw new Error(`${filename} missing and no URL configured`);
  console.log(`  Downloading ${filename}...`);
  execSync(`curl -fSL -o "${fp}" "${url}"`, { stdio: 'inherit', timeout: 300000 });
}

async function init() {
  console.log('Checking data files...');
  for (const [f, url] of Object.entries(FILES)) await downloadIfMissing(f, url);

  console.log('Loading model...');
  const ort = require('onnxruntime-node');
  session = await ort.InferenceSession.create(path.join(__dirname, 'model.onnx'));
  meta = JSON.parse(fs.readFileSync(path.join(__dirname, 'vs-meta.json'), 'utf8'));
  const bin = fs.readFileSync(path.join(__dirname, 'vs-embeddings.bin'));
  embeddings = new Float32Array(meta.length * 512);
  for (let i = 0; i < meta.length * 512; i++) {
    const h = bin.readUInt16LE(i * 2);
    const s = (h & 0x8000) >> 15, e = (h & 0x7C00) >> 10, m = h & 0x03FF;
    embeddings[i] = e === 0 ? (s ? -1 : 1) * 2 ** -14 * (m / 1024)
      : e === 31 ? (m ? NaN : (s ? -Infinity : Infinity))
      : (s ? -1 : 1) * 2 ** (e - 15) * (1 + m / 1024);
  }
  console.log(`Ready: ${meta.length} products, port ${PORT}`);
}

async function embed(buf) {
  const sharp = require('sharp');
  const ort = require('onnxruntime-node');
  const { data } = await sharp(buf).resize(224, 224, { fit: 'cover' }).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const f = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++)
    for (let c = 0; c < 3; c++) f[c * 224 * 224 + i] = (data[i * 3 + c] / 255 - MEAN[c]) / STD[c];
  const res = await session.run({ pixel_values: new ort.Tensor('float32', f, [1, 3, 224, 224]) });
  const emb = Array.from(Object.values(res)[0].data);
  const norm = Math.sqrt(emb.reduce((s, v) => s + v * v, 0));
  return emb.map(v => v / norm);
}

function search(query, topK = 20) {
  const scores = new Float32Array(meta.length);
  for (let i = 0; i < meta.length; i++) {
    let dot = 0, off = i * 512;
    for (let j = 0; j < 512; j++) dot += query[j] * embeddings[off + j];
    scores[i] = dot;
  }
  return [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, topK)
    .map(i => ({ id: meta[i][0], handle: meta[i][1], title: meta[i][2], img: meta[i][3], score: (scores[i] * 100).toFixed(1) }));
}

const server = http.createServer(async (req, res) => {
  const origin = req.headers.origin || '';
  if (ALLOWED.some(a => origin.startsWith(a))) res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.end();
  if (req.method === 'GET' && req.url === '/') {
    res.setHeader('Content-Type', 'text/html');
    return fs.createReadStream(path.join(__dirname, 'visual-search.html')).pipe(res);
  }
  if (req.method === 'GET' && req.url === '/health') {
    return res.end(JSON.stringify({ ok: true, products: meta.length }));
  }
  if (req.method === 'POST' && req.url === '/search') {
    const chunks = []; let size = 0;
    for await (const c of req) { size += c.length; if (size > 10e6) { res.statusCode = 413; return res.end('Too large'); } chunks.push(c); }
    try {
      const t0 = Date.now();
      const results = search(await embed(Buffer.concat(chunks)));
      console.log(`Search: ${Date.now() - t0}ms`);
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(results));
    } catch (e) { res.statusCode = 500; res.end(JSON.stringify({ error: e.message })); }
    return;
  }
  res.statusCode = 404; res.end('Not found');
});

init().then(() => server.listen(PORT));
