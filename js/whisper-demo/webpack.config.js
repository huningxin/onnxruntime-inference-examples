
import path from 'path';
import { fileURLToPath } from 'url';

import CopyPlugin from 'copy-webpack-plugin';
import autoprefixer from 'autoprefixer';
import HtmlWebpackPlugin from 'html-webpack-plugin';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const config = {
    mode: 'development',
    devtool: 'inline-source-map',
    entry: {
        content: './content.js',
        polyfill: './polyfill.js',
        background: './background.js',
        streaming_processor: './streaming_processor.js',
        options: './options.js'
    },
    output: {
        path: path.resolve(__dirname, 'build'),
        filename: '[name].js',
    },
    plugins: [
        new HtmlWebpackPlugin({ template: './options.html' }),
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
                },
                {
                    // Options UI
                    from: "options.html",
                    to: "." // Copies to build folder
                }
            ],
        })
    ],
    module: {
        rules: [
        {
            test: /\.(scss)$/,
            use: [
            {
                // Adds CSS to the DOM by injecting a `<style>` tag
                loader: 'style-loader'
            },
            {
                // Interprets `@import` and `url()` like `import/require()` and will resolve them
                loader: 'css-loader'
            },
            {
                // Loader for webpack to process CSS with PostCSS
                loader: 'postcss-loader',
                options: {
                postcssOptions: {
                    plugins: [
                    autoprefixer
                    ]
                }
                }
            },
            {
                // Loads a SASS/SCSS file and compiles it to CSS
                loader: 'sass-loader'
            }
            ]
        }
    ]
  }
};

export default config;
