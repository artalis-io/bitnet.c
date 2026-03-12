// Web Worker for non-blocking BitNet inference
// Communicates with main thread via postMessage

importScripts('bitnet.js');

let bitnet = null;
let initialized = false;

async function init(modelUrl) {
    try {
        postMessage({ type: 'status', message: 'Loading WASM module...' });
        const module = await BitNet();

        postMessage({ type: 'status', message: `Fetching model: ${modelUrl}` });
        const response = await fetch(modelUrl);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const totalBytes = parseInt(response.headers.get('content-length') || '0');
        const reader = response.body.getReader();
        const chunks = [];
        let loaded = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            if (totalBytes > 0) {
                postMessage({
                    type: 'progress',
                    loaded,
                    total: totalBytes,
                    percent: Math.round(loaded / totalBytes * 100)
                });
            }
        }

        // Concatenate chunks
        const modelData = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
            modelData.set(chunk, offset);
            offset += chunk.length;
        }

        postMessage({ type: 'status', message: 'Initializing model...' });

        // Copy to WASM heap
        const ptr = module._malloc(modelData.length);
        module.HEAPU8.set(modelData, ptr);

        // Initialize
        const initFn = module.cwrap('bitnet_init', 'number', ['number', 'number']);
        const rc = initFn(ptr, modelData.length);
        if (rc !== 0) throw new Error('bitnet_init failed');

        bitnet = {
            module,
            dataPtr: ptr,
            forward: module.cwrap('bitnet_forward_token', 'number', ['number', 'number']),
            sample: module.cwrap('bitnet_sample', 'number', ['number', 'number']),
            encode: module.cwrap('bitnet_encode', 'number', ['number', 'number', 'number', 'number']),
            decode: module.cwrap('bitnet_decode', 'string', ['number']),
            vocabSize: module.cwrap('bitnet_vocab_size', 'number', []),
            bosId: module.cwrap('bitnet_bos_id', 'number', []),
            eosId: module.cwrap('bitnet_eos_id', 'number', []),
            free: module.cwrap('bitnet_free', null, []),
        };

        initialized = true;
        postMessage({ type: 'ready', vocabSize: bitnet.vocabSize() });
    } catch (e) {
        postMessage({ type: 'error', message: e.message });
    }
}

function encodePrompt(text) {
    const maxTokens = text.length + 16;
    const bufPtr = bitnet.module._malloc(maxTokens * 4);
    const textPtr = bitnet.module._malloc(text.length + 1);

    // Write text to WASM memory
    const textBytes = new TextEncoder().encode(text + '\0');
    bitnet.module.HEAPU8.set(textBytes, textPtr);

    const n = bitnet.encode(textPtr, 1, bufPtr, maxTokens);
    const tokens = [];
    for (let i = 0; i < n; i++) {
        tokens.push(bitnet.module.HEAPF32[(bufPtr >> 2) + i] | 0);
        // Actually should read as int32:
    }

    // Read as Int32Array
    const int32View = new Int32Array(bitnet.module.HEAPU8.buffer, bufPtr, n);
    const result = Array.from(int32View);

    bitnet.module._free(bufPtr);
    bitnet.module._free(textPtr);

    return result;
}

function generate(prompt, maxTokens, temperature, topp) {
    if (!initialized) {
        postMessage({ type: 'error', message: 'Not initialized' });
        return;
    }

    try {
        const tokens = encodePrompt(prompt);
        postMessage({ type: 'status', message: `Prompt: ${tokens.length} tokens` });

        let token = tokens[0];
        let pos = 0;
        const eosId = bitnet.eosId();

        for (let i = 0; i < tokens.length + maxTokens; i++) {
            bitnet.forward(token, pos);

            let next;
            if (i < tokens.length - 1) {
                next = tokens[i + 1];
            } else {
                next = bitnet.sample(temperature, topp);

                if (next === eosId) {
                    postMessage({ type: 'done' });
                    return;
                }

                const piece = bitnet.decode(next);
                postMessage({ type: 'token', text: piece, id: next });
            }

            token = next;
            pos++;
        }

        postMessage({ type: 'done' });
    } catch (e) {
        postMessage({ type: 'error', message: e.message });
    }
}

// Message handler
onmessage = function(e) {
    const { type, ...params } = e.data;

    switch (type) {
        case 'init':
            init(params.modelUrl);
            break;
        case 'generate':
            generate(
                params.prompt || 'Hello',
                params.maxTokens || 256,
                params.temperature || 0.0,
                params.topp || 0.9
            );
            break;
        case 'free':
            if (bitnet) {
                bitnet.free();
                if (bitnet.dataPtr) bitnet.module._free(bitnet.dataPtr);
                initialized = false;
            }
            break;
    }
};
