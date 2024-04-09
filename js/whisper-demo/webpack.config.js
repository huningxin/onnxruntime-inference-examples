
import path from 'path';
import { fileURLToPath } from 'url';

import CopyPlugin from 'copy-webpack-plugin';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const config = {
    mode: 'development',
    devtool: 'inline-source-map',
    entry: {
        main: './main.js',
        content: './content.js',
        streaming_processor: './streaming_processor.js',
        polyfill: './polyfill.js'
    },
    output: {
        path: path.resolve(__dirname, 'build'),
        filename: '[name].js',
    },
    plugins: [
        new CopyPlugin({
            patterns: [
                {
                    from: "models",
                    to: "models" // Copies to build folder
                },
                {
                    from: "manifest.json",
                    to: "." // Copies to build folder
                },
                {
                    // Use copy plugin to copy *.wasm to output folder.
                    from: 'node_modules/onnxruntime-web/dist/*.wasm',
                    to: '[name][ext]'
                }
            ],
        })
    ],
};

export default config;
