# Quick Start - Web (using script tag)

This example is a demonstration of basic usage of ONNX Runtime Web, using script tag in HTML.

Using a `<script>` tag is a simple way to consume a published JavaScript library. See also [Quick Start - Web (using bundler)](../quick-start_onnxruntime-web-bundler) for an example of using bundler.

In this example, we load onnxruntime, create an inference session with a simple model, feed input, get output as result and write it to the HTML page. All functions are called in their basic form.

## Usage

1. use NPM package `light-server` to serve the current folder at http://localhost:8080/
   ```sh
   npx light-server -s . -p 8080
   ```

2. open your browser and navigate to the URL.

## Wasm example

This example emulates the session inference path: JS (index_wasm.html) -> Wasm (inference.cpp) -> ORT JS API -> ORT Wasm.

1. Compile inference.cpp by
   ```sh
   emcc inference.cpp -o inference.js -s EXPORTED_FUNCTIONS="['_run_session']" -s ASYNCIFY=1 -s MODULARIZE=1 -s EXPORT_NAME="createModule"
   ```

2. Start a server as above
   ```sh
   npx light-server -s . -p 8080
   ```

3. Open your and navigate to http://localhost:8080/index_wasm.html
