// Runs Tachyon (WebAssembly) off the page's thread: loads a model's files
// into the module's memory, then streams replies token by token. The module
// exports plain functions (see src/web.rs); no bindings library.

let wasm = null;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

// Memory can grow during any call, which detaches old views: make a fresh
// view for every access.
const bytesAt = (ptr, len) => new Uint8Array(wasm.memory.buffer, ptr, len);
const output = () => decoder.decode(bytesAt(wasm.tachyon_out(), wasm.tachyon_out_len()));

function put(bytes) {
  const ptr = wasm.tachyon_alloc(bytes.length);
  bytesAt(ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

// Downloads a file straight into wasm memory (a 1 GB checkpoint is never
// held twice). Returns [ptr, len], or [0, 0] for an optional missing file.
async function fetchInto(url, label, optional = false) {
  const response = await fetch(url);
  if (!response.ok) {
    if (optional) return [0, 0];
    throw new Error(`${url}: HTTP ${response.status}`);
  }
  const total = Number(response.headers.get('content-length')) || 0;
  if (!total) return put(new Uint8Array(await response.arrayBuffer()));
  const ptr = wasm.tachyon_alloc(total);
  const reader = response.body.getReader();
  let got = 0;
  let shown = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bytesAt(ptr + got, value.length).set(value);
    got += value.length;
    if (got - shown > total / 100 || got === total) {
      shown = got;
      postMessage({ type: 'progress', label, got, total });
    }
  }
  if (got !== total) throw new Error(`${url}: got ${got} of ${total} bytes`);
  return [ptr, total];
}

async function load({ base, format }) {
  if (!wasm) {
    const { instance } = await WebAssembly.instantiate(
      await (await fetch('tachyon.wasm')).arrayBuffer(),
      {},
    );
    wasm = instance.exports;
  }
  const url = (name) => new URL(name, base).href;
  const config = await fetchInto(url('config.json'), 'config');
  const generation = await fetchInto(url('generation_config.json'), 'generation config', true);
  const tokenizer = await fetchInto(url('tokenizer.json'), 'tokenizer');
  const weights = await fetchInto(url('model.safetensors'), 'weights');
  postMessage({ type: 'progress', label: 'converting weights', got: 0, total: 0 });
  const started = performance.now();
  if (wasm.tachyon_load(...config, ...generation, ...tokenizer, ...weights, format) !== 0) {
    throw new Error(output());
  }
  postMessage({ type: 'ready', seconds: (performance.now() - started) / 1000 });
}

function chat({ text, maxTokens, temperature }) {
  const seed = Math.floor(Math.random() * 2 ** 31);
  const started = performance.now();
  if (wasm.tachyon_start(...put(encoder.encode(text)), maxTokens, temperature, seed) !== 0) {
    throw new Error(output());
  }
  const first = performance.now();
  let tokens = 0;
  while (wasm.tachyon_next() >= 0) {
    tokens += 1;
    const piece = output();
    if (piece) postMessage({ type: 'token', text: piece });
  }
  const end = performance.now();
  postMessage({
    type: 'done',
    tokens,
    prefillSeconds: (first - started) / 1000,
    decodeSeconds: (end - first) / 1000,
  });
}

onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') await load(data);
    else if (data.type === 'chat') chat(data);
    else if (data.type === 'reset') wasm?.tachyon_reset();
  } catch (e) {
    let message = String(e.message || e);
    if (e instanceof WebAssembly.RuntimeError && wasm?.tachyon_panic_len()) {
      message = decoder.decode(bytesAt(wasm.tachyon_panic(), wasm.tachyon_panic_len()));
    }
    postMessage({ type: 'error', message });
  }
};
