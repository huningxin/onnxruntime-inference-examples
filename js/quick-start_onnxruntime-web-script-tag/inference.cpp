#include <emscripten/emscripten.h>
#include <iostream>
#include <numeric>

EM_ASYNC_JS(void, load_model_async, (), {
  console.log("[JS] Starting async ONNX model loading...");

  // Load ONNX Runtime Web if not already loaded
  if (typeof ort === 'undefined') {
      console.error("ORT Web not loaded");
      return;
  }

  try {
    // create a new session and load the specific model.
    //
    // the model in this example contains a single MatMul node
    // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
    // it has 1 output: 'c'(float32, 3x3)
    window.session = await ort.InferenceSession.create('./model.onnx');

    } catch (e) {
    console.log(`failed to inference ONNX model: ${e}.`);
  }
});

EM_ASYNC_JS(void, run_model_async, (void* bufferA, size_t bufferALength, void* bufferB, size_t bufferBLength, void* bufferC, size_t bufferCLength), {
  console.log("[JS] Starting async ONNX model inference...");

  // Load ONNX Runtime Web if not already loaded
  if (typeof ort === 'undefined') {
      console.error("ORT Web not loaded");
      return;
  }

  try {
    // prepare inputs from C++ vector buffers
    const dataA = new Float32Array(Module["HEAPU8"].buffer, bufferA, bufferALength);
    console.log(`[JS] data of tensor 'a': ${dataA}`);
    const dataB = new Float32Array(Module["HEAPU8"].buffer, bufferB, bufferBLength);
    console.log(`[JS] data of tensor 'b': ${dataB}`);
    const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
    const tensorB = new ort.Tensor('float32', dataB, [4, 3]);

    // prepare feeds. use model input names as keys.
    const feeds = { a: tensorA, b: tensorB };

    // feed inputs and run
    const results = await session.run(feeds);

    // read from results
    const dataC = results.c.data;
    console.log(`[JS] data of result tensor 'c': ${dataC}`);

    if (dataC.length != bufferCLength) {
      console.log(`[JS] the dataC.length ${dataC.length} does not match bufferCLength ${bufferCLength}`);
    }

    // set data back to C++ vector buffer
    Module["HEAPF32"].set(dataC, bufferC >> 2);
  } catch (e) {
    console.log(`[JS] failed to inference ONNX model: ${e}.`);
  }
});

void print_vector(const std::vector<float>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

extern "C" {
  EMSCRIPTEN_KEEPALIVE
  void run_session() {
      std::cout << "[C++] Before loading session..." << std::endl;
      load_model_async();
      std::cout << "[C++] After loading session..." << std::endl;
      // inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
      std::vector<float> dataA(12);
      std::vector<float> dataB(12);
      // output: 'c'(float32, 3x3)
      std::vector<float> dataC(9, 0);
      for (int i = 0; i < 10; i++) {
        std::cout << "[C++] Prepare inputs..." << std::endl;
        std::iota(dataA.begin(), dataA.end(), 1 + i);
        std::cout << "[C++] dataA ";
        print_vector(dataA);
        std::iota(dataB.begin(), dataB.end(), 10 + i);
        std::cout << "[C++] dataB ";
        print_vector(dataB);
        std::cout << "[C++] Before running session..." << std::endl;
        run_model_async(dataA.data(), dataA.size(), dataB.data(), dataB.size(), dataC.data(), dataC.size());
        std::cout << "[C++] After running session..." << std::endl;
        std::cout << "[C++] dataC ";
        print_vector(dataC);
      }
  }
}
