# WebNN Whisper browser extension

An example project to show how to polyfill [Web Speech API](https://wicg.github.io/speech-api/) with WebNN Whisper models.

## Getting Started
1. Clone the repo and enter the project directory.
1. Download Whisper models and put them into `models` folder:
   1. https://huggingface.co/lwanming/whisper-base-static-shape/blob/main/whisper_base_encoder_lm_fp16_layernorm_gelu.onnx
   1. https://huggingface.co/lwanming/whisper-base-static-shape/blob/main/whisper_base_decoder_static_non_kvcache_lm_fp16_layernorm_gelu_4dmask.onnx
   1. https://huggingface.co/lwanming/whisper-base-static-shape/blob/main/whisper_base_decoder_static_kvcache_128_lm_fp16_layernorm_gelu_4dmask.onnx

1. Install the necessary dependencies:
    ```bash
    npm install 
    ```

1. Build the project:
    ```bash
    npm run build 
    ```

1. Add the extension to your browser. To do this, go to `chrome://extensions/`, enable developer mode (top right), and click "Load unpacked". Select the `build` directory from the dialog which appears and click "Select Folder".

1. That's it! You should now be able to open the web apps using Web Speech API, i.e. https://www.google.com/intl/en/chrome/demos/speech.html, polyfilled by WebNN Whisper models.