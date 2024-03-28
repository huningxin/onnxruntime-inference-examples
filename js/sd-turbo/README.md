# Running Stable Diffusion Turbo in your Browser with WebGPU and ONNX Runtime Web

This repository contains an example of running [Stability AI's Stable Diffusion Turbo Model](https://huggingface.co/stabilityai/sd-turbo) in yor browser using [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) with WebGPU.

You can try out the live demo [here](https://guschmue.github.io/ort-webgpu/sd-turbo/index.html).

## Model Overview

SD-Turbo is a fast generative text-to-image model that can synthesize photorealistic images from a text prompt in a single network evaluation. This is a reseach oriented model to study small, distilled text-to-image models.
For more details checkout [Stabilities research report](https://stability.ai/research/adversarial-diffusion-distillation). 

### The ONNX Model

The model used in this project is hosted on [Hugging Face](https://huggingface.co/lwanming/sd-turbo-ort-web). 


### Running the Project

Start a web server to serve the current folder at http://localhost:8888/. To start the server, e.g. run:

```sh
python -m http.server
```
