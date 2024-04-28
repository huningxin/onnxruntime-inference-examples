//'@xenova/transformers';
import { get_new_tokens } from './generation_utils.js';
import { attention_mask_update, cache_update } from './post_processing.js';
import {log, getModelOPFS, convertToFloat32Array, convertToUint16Array} from './utils.js';

const FP16Bytes = 2;
const Int32Bytes = 4;
const Int64Bytes = 8;

// wrapper around onnxruntime and model
export class Whisper {
    constructor(url, provider, deviceType = 'gpu', dataType, ort, AutoProcessor, AutoTokenizer, verbose = false) {
        this.url = url;
        this.provider = provider;
        this.deviceType = deviceType;
        this.dataType = dataType;
        this.ort = ort;
        this.ort.env.wasm.simd = true;
        this.AutoProcessor = AutoProcessor;
        this.AutoTokenizer = AutoTokenizer;
        this.verbose = verbose;
        
        // Hack to get MLContext after WebNN EP initialized.
        this.context = wnn_context_;
        this.models = {
            'encoder': { url: 'whisper_base_encoder_lm.onnx', sess: null, graph: null },
            'decoder': { url: 'whisper_base_decoder_static_non_kvcache_lm.onnx', sess: null, graph: null},
            'decoder_cached': { url: 'whisper_base_decoder_static_kvcache_128_lm.onnx', sess: null, graph: null},
        };

        this.max_sequence_length = 128;
        // No. of tokens to be used for decoder 1st inference
        this.num_init_tokens = 4;
        // Whisper was trained using 16000 Hz as sampling rate, fixing this value for dataset preparation
        this.sampling_rate = 16000;
        this.processor = null;
        this.tokenizer = null;
    }

    async create_whisper_processor() {
        // processor contains feature extractor
        this.processor = await this.AutoProcessor.from_pretrained('openai/whisper-base');
    }

    async create_whisper_tokenizer() {
        // processor contains feature extractor
        this.tokenizer = await this.AutoTokenizer.from_pretrained('openai/whisper-base', { config: { do_normalize: true } });
    }

    async create_ort_sessions() {
        const options = {
            executionProviders: [{
                name: this.provider,
                deviceType: this.deviceType,
            }],
        };

        for (let name of Object.keys(this.models)) {
            try {
                let url = this.url + this.models[name]['url'];
                if (this.dataType == 'float16') {
                    url = url.replace('.onnx', '_fp16_layernorm.onnx');
                } else {
                    url = url.replace('.onnx', '_layernorm.onnx');
                }
                if (!url.startsWith('chrome-extension://')) {
                    const modelBuffer = await getModelOPFS(`${name}_${this.dataType}`, url, false);
                    this.models[name]['sess'] = await this.ort.InferenceSession.create(modelBuffer, options);
                } else {
                    this.models[name]['sess'] = await this.ort.InferenceSession.create(url, options);
                }
                // Hack to get the MLGraph in the created session.
                this.models[name]['graph'] = wnn_graph_;
                log(`Model ${url} loaded`);
            } catch (e) {
                log(`Error: ${e}`);
            }
        };

        this.models['encoder']['inputs'] = {'input_features': this.context.createBuffer({size: 1 * 80 * 3000 * FP16Bytes})};
        this.models['encoder']['outputs'] = {'last_hidden_state': this.context.createBuffer({size: 1 * 1500 * 512 * FP16Bytes})};

        // MLBuffers for non KV cache decoder
        this.models['decoder']['inputs'] = {
            'input_ids': this.context.createBuffer({size: 1 * 4 * Int32Bytes}),
            'attention_mask': this.context.createBuffer({size: 1 * 4 * Int64Bytes}),
            // Use encoder's last hidden_state buffer.
            'encoder_hidden_states': this.models['encoder']['outputs'].last_hidden_state,
        };
        this.models['decoder']['outputs'] = {
            'logits': this.context.createBuffer({size: 1 * 4 * 51865 * FP16Bytes}),
        };
        for (let i = 0; i < 6; ++i) {
            this.models['decoder']['outputs'][`present_key_values.${i}.encoder.key`] = this.context.createBuffer({size: 1 * 8 * 1500 * 64 * FP16Bytes});
            this.models['decoder']['outputs'][`present_key_values.${i}.encoder.value`] = this.context.createBuffer({size: 1 * 8 * 1500 * 64 * FP16Bytes});
            this.models['decoder']['outputs'][`present_key_values.${i}.decoder.key`] = this.context.createBuffer({size: 1 * 8 * 4 * 64 * FP16Bytes});
            this.models['decoder']['outputs'][`present_key_values.${i}.decoder.value`] = this.context.createBuffer({size: 1 * 8 * 4 * 64 * FP16Bytes});
        }

        // MLBuffers for KV cache decoder
        this.models['decoder_cached']['inputs'] = {
            'input_ids': this.context.createBuffer({size: 1 * 1 * Int32Bytes}),
            'attention_mask': this.context.createBuffer({size: 1 * 128 * Int64Bytes}),
            'position_ids': this.context.createBuffer({size: 1 * Int32Bytes}),
        };
        for (let i = 0; i < 6; ++i) {
            this.models['decoder_cached']['inputs'][`past_key_values.${i}.encoder.key`] = this.models['decoder']['outputs'][`present_key_values.${i}.encoder.key`];
            this.models['decoder_cached']['inputs'][`past_key_values.${i}.encoder.value`] = this.models['decoder']['outputs'][`present_key_values.${i}.encoder.value`];
            this.models['decoder_cached']['inputs'][`past_key_values.${i}.decoder.key`] = this.context.createBuffer({size: 1 * 8 * 127 * 64 * FP16Bytes});
            this.models['decoder_cached']['inputs'][`past_key_values.${i}.decoder.value`] = this.context.createBuffer({size: 1 * 8 * 127 * 64 * FP16Bytes});
        }
        this.models['decoder_cached']['outputs'] = {
            // Reuse non KV cache decoder's logits buffer.
            'logits': this.models['decoder']['outputs'].logits,
        };
        for (let i = 0; i < 6; ++i) {
            this.models['decoder_cached']['outputs'][`present_key_values.${i}.decoder.key`] = this.context.createBuffer({size: 1 * 8 * 1 * 64 * FP16Bytes});
            this.models['decoder_cached']['outputs'][`present_key_values.${i}.decoder.value`] = this.context.createBuffer({size: 1 * 8 * 1 * 64 * FP16Bytes});
        }

        await this.build_pad_cache_graph();
        await this.build_increment_position_ids_graphs();
        await this.build_update_attention_mask_graphs();
    }

    async build_pad_cache_graph() {
        const builder = new MLGraphBuilder(this.context);
        const outputs = {};
        for (let i = 0; i < 6; ++i) {
            const key = builder.input(`present_key_values.${i}.decoder.key`, {dataType: 'float16', dimensions: [1, 8, 4, 64]})
            outputs[`past_key_values.${i}.decoder.key`] = builder.pad(key, [0, 0, 0, 0], [0, 0, 123, 0]);
            const value = builder.input(`present_key_values.${i}.decoder.value`, {dataType: 'float16', dimensions: [1, 8, 4, 64]})
            outputs[`past_key_values.${i}.decoder.value`] = builder.pad(value, [0, 0, 0, 0], [0, 0, 123, 0]);
        }
        this.models['pad_cache'] = {};
        this.models['pad_cache'].graph = await builder.build(outputs);
        this.models['pad_cache'].inputs = {};
        this.models['pad_cache'].outputs = {};
        for (let i = 0; i < 6; ++i) {
            this.models['pad_cache'].inputs[`present_key_values.${i}.decoder.key`] = this.models['decoder']['outputs'][`present_key_values.${i}.decoder.key`];
            this.models['pad_cache'].inputs[`present_key_values.${i}.decoder.value`] = this.models['decoder']['outputs'][`present_key_values.${i}.decoder.value`];
            this.models['pad_cache'].outputs[`past_key_values.${i}.decoder.key`] = this.models['decoder_cached']['inputs'][`past_key_values.${i}.decoder.key`];
            this.models['pad_cache'].outputs[`past_key_values.${i}.decoder.value`] = this.models['decoder_cached']['inputs'][`past_key_values.${i}.decoder.value`];
        }
    }

    dispatch_graph(name) {
        this.context.dispatch(this.models.name.graph, this.models.name.inputs, this.models.name.outputs);
    }

    async build_increment_position_ids_graphs() {
        const builder = new MLGraphBuilder(this.context);

        // temp_position_ids = position_ids + 1
        const position_ids_desc = {dataType: 'int32', dimensions: [1]};
        const position_ids = builder.input('position_ids', position_ids_desc);
        const one = builder.constant({dataType: 'int32', dimensions: [1]}, new Int32Array([1]));
        const temp_position_ids = builder.add(position_ids, one);
        this.models['increment_position_ids'] = {};
        this.models['increment_position_ids'].graph = await builder.build({temp_position_ids});
        this.models['increment_position_ids'].inputs = {'position_ids': this.models['decoder_cached']['inputs'].position_ids};
        this.models['increment_position_ids'].outputs = {'temp_position_ids': this.context.createBuffer({size: 1 * Int32Bytes})};

        // position_ids = temp_position_ids
        const input = builder.input('input', position_ids_desc);
        const output = builder.identity(input);
        this.models['copy_position_ids'] = {};
        this.models['copy_position_ids'].graph = await builder.build({output});
        this.models['copy_position_ids'].inputs = {'input': this.models['increment_position_ids'].outputs.temp_position_ids};
        this.models['copy_position_ids'].outputs = {'output': this.models['decoder_cached']['inputs'].position_ids};
    }

    dispatch_increment_position_ids_graphs() {
        this.dispatch_graph('increment_position_ids');
        this.dispatch_graph('copy_position_ids')
    }

    async build_update_attention_mask_graphs() {
        const builder = new MLGraphBuilder(this.context);

        // temp_attention_mask = attention_mask;
        // temp_attention_mask[position_ids - 1] = 1;
        const attention_mask_desc = {dataType: 'int64', dimensions: [128]};
        const attention_mask = builder.input('attention_mask', attention_mask_desc);
        const position_ids = builder.input('position_ids', {dataType: 'int32', dimensions: [1, 1]});
        const one = builder.constant({dataType: 'int32', dimensions: [1, 1]}, new Int32Array([1]));
        const indices = builder.sub(position_ids, one);
        const updates = builder.constant({dataType: 'int64', dimensions: [1]}, new BigInt64Array([1n]));
        const temp_attention_mask = builder.scatterNd(attention_mask, indices, updates);
        this.models['update_attention_mask'] = {};
        this.models['update_attention_mask'].graph = await builder.build({temp_attention_mask});
        this.models['update_attention_mask'].inputs = {
            'attention_mask': this.models['decoder_cached']['inputs'].attention_mask,
            'position_ids': this.models['decoder_cached']['inputs'].position_ids};
        this.models['update_attention_mask'].outputs = {'temp_attention_mask': this.context.createBuffer({size: 1 * 128 * Int64Bytes})};

        // attention_mask = temp_attention_mask;
        const input = builder.input('input', attention_mask_desc);
        const output = builder.identity(input);
        this.models['copy_attention_mask'] = {};
        this.models['copy_attention_mask'].graph = await builder.build({output});
        this.models['copy_attention_mask'].inputs = {'input': this.models['update_attention_mask'].outputs.temp_attention_mask};
        this.models['copy_attention_mask'].outputs = {'output': this.models['decoder_cached']['inputs'].attention_mask};
    }

    dispatch_update_attention_mask_graphs() {
        this.dispatch_graph('update_attention_mask');
        this.dispatch_graph('copy_attention_mask');
    }

    async build_update_cache_graphs() {
        const builder = new MLGraphBuilder(this.context);

        // temp_past_key_values = past_key_values;
        // past_key_values[position_ids - 1, head, hidden_dimension] = present_key_values;
        const position_ids = builder.input('position_ids', {dataType: 'int32', dimensions: [1, 1]});
        const one = builder.constant({dataType: 'int32', dimensions: [1, 1]}, new Int32Array([1]));
        const indices = builder.sub(position_ids, one);

        const outputs = {};
        for (let i = 0; i < 6; ++i) {
            // Transpose HSD (batch = 1) [head, squence_length, hidden_dimension] to SHD, so the scatter can only update the S axis.
            const perm = [1, 0, 2];
            const past_kv_desc = {dataType: 'float16', dimensions: [8, 127, 64]};
            const pask_key = builder.transpose(builder.input(`past_key_values.${i}.decoder.key`, past_kv_desc), perm);
            const past_value = builder.transpose(builder.input(`past_key_values.${i}.decoder.value`, past_kv_desc), perm);
            const present_kv_desc = {dataType: 'float16', dimensions: [8, 1, 64]};
            const present_key = builder.transpose(builder.input(`present_key_values.${i}.decoder.key`, present_kv_desc), perm);
            const present_value = builder.transpose(builder.input(`present_key_values.${i}.decoder.value`, present_kv_desc), perm);

            outputs[`past_key_values.${i}.decoder.key`] = builder.transpose(builder.scatterNd(pask_key, indices, present_key), perm);
            outputs[`past_key_values.${i}.decoder.value`] = builder.transpose(builder.scatterNd(past_value, indices, present_value), perm);
        }

        this.models['update_cache'] = {};
        this.models['update_cache'].graph = await builder.build(outputs);
        this.models['update_cache'].inputs = {'position_ids': this.models['decoder_cached']['inputs'].position_ids};
        this.models['update_cache'].outputs = {};
        for (let i = 0; i < 6; ++i) {
            this.models['update_cache'].inputs[`past_key_values.${i}.decoder.key`] = this.models['decoder_cached']['inputs'][`past_key_values.${i}.decoder.key`];
            this.models['update_cache'].inputs[`past_key_values.${i}.decoder.value`] = this.models['decoder_cached']['inputs'][`past_key_values.${i}.decoder.value`];
            this.models['update_cache'].inputs[`present_key_values.${i}.decoder.key`] = this.models['decoder_cached']['outputs'][`present_key_values.${i}.decoder.key`];
            this.models['update_cache'].inputs[`present_key_values.${i}.decoder.value`] = this.models['decoder_cached']['outputs'][`present_key_values.${i}.decoder.value`];
            this.models['update_cache'].outputs[`past_key_values.${i}.decoder.key`] = this.context.createBuffer({size: 8 * 127 * 64});
            this.models['update_cache'].outputs[`past_key_values.${i}.decoder.value`] = this.context.createBuffer({size: 8 * 127 * 64});
        }

        // past_key_values = temp_past_key_values;
        const copy_outputs = {};
        for (let i = 0; i < 6; ++i) {
            // Transpose HSD (batch = 1) [head, squence_length, hidden_dimension] to SHD, so the scatter can only update the S axis.
            const past_kv_desc = {dataType: 'float16', dimensions: [8, 127, 64]};
            const temp_pask_key = builder.input(`temp_past_key_values.${i}.decoder.key`, past_kv_desc);
            const temp_past_value = builder.input(`temp_past_key_values.${i}.decoder.value`, past_kv_desc);
            outputs[`past_key_values.${i}.decoder.key`] = builder.identity(temp_pask_key);
            outputs[`past_key_values.${i}.decoder.value`] = builder.identity(temp_pask_value);
        }
        this.models['copy_cache'] = {};
        this.models['copy_cache'].graph = await builder.build(copy_outputs);
        this.models['copy_cache'].inputs = {};
        this.models['copy_cache'].outputs = {};
        for (let i = 0; i < 6; ++i) {
            this.models['copy_cache'].inputs[`temp_past_key_values.${i}.decoder.key`] = this.models['update_cache'].outputs[`past_key_values.${i}.decoder.key`];
            this.models['copy_cache'].inputs[`temp_past_key_values.${i}.decoder.value`] = this.models['update_cache'].outputs[`past_key_values.${i}.decoder.value`];
            this.models['copy_cache'].outputs[`past_key_values.${i}.decoder.key`] = this.models['decoder_cached']['inputs'][`past_key_values.${i}.decoder.key`];
            this.models['copy_cache'].outputs[`past_key_values.${i}.decoder.value`] = this.models['decoder_cached']['inputs'][`past_key_values.${i}.decoder.value`];
        }
    }

    dispatch_update_cache_graphs() {
        this.dispatch_graph('update_cache');
        this.dispatch_graph('copy_cache');
    }

    async run(audio_data) {
        // -----------------------------------FEATURE EXTRACTION-----------------------------------------
        // const audio = await read_audio('https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac', 16000);
        // const audio = await read_audio(audio_data, sampling_rate);
        let start = performance.now();
        const { input_features } = await this.processor(audio_data);
        // -----------------------------------ENCODER INFERENCE-----------------------------------------
        // run encoder to get output
        let input_features_data;
        if (this.dataType == 'float16') {
            input_features_data = convertToUint16Array(input_features.data);
        }
        if (this.verbose) {
            console.log(`  pre-processing time: ${(performance.now() - start).toFixed(2)} ms`);
        }
        start = performance.now();
        
        // Upload input data and dispatch graph
        this.context.writeBuffer(this.models['encoder'].inputs.input_features, input_features_data, input_features_data.length);
        this.context.dispatch(this.models['encoder'].graph, this.models['encoder'].inputs, this.models['encoder'].outputs);

        if (this.verbose) {
            console.log(`  encoder inference time: ${(performance.now() - start).toFixed(2)} ms`);
        }
        start = performance.now();
        // -----------------------------------DECODER 1ST INFERENCE-----------------------------------------
        // create list of tokens for english language and transcribe task, no need of time stamps
        // TODO: CHANGE FROM HARDCODED VALUES
        let tokens = [50258, 50259, 50359, 50363];
        // let tokens = [50258, 50259, 50359, 50364]; // keep timestep token
        const attention_mask = [1, 1, 1, 1];

        if (this.verbose) {
            console.log(`  non-kv cache decoder input preparation time: ${(performance.now() - start).toFixed(2)} ms`);
        }
        start = performance.now();
        // run the first inference which generates SA and CA KV cache
        // const decoder_output = await this.models['decoder']['sess'].run(decoder_input);

        // Upload tokens and attention_mask and dispatch the decoder
        this.context.writeBuffer(this.models['decoder'].inputs.input_ids, new Int32Array(tokens), tokens.length);
        this.context.writeBuffer(this.models['decoder'].inputs.attention_mask, new Int32Array(attention_mask), attention_mask.length);
        this.context.dispatch(this.models['decoder'].graph, this.models['decoder'].inputs, this.models['decoder'].outputs);

        if (this.verbose) {
            console.log(`  non-kv cache decoder inference time: ${(performance.now() - start).toFixed(2)} ms`);
        }
        start = performance.now();
        // let logits = decoder_output['logits']['cpuData'];
        const logitsBuffer = await this.context.readBuffer(this.models['decoder'].outputs.logits);
        let logits = new Uint16Array(logitsBuffer);

        if (this.dataType == 'float16') {
            logits = convertToFloat32Array(logits);
        }
        // find out the token with highest probability, cast INT64 to INT32
        const new_token = get_new_tokens(logits, [1, 4, 51865]);

        // add token to final buffer
        tokens = tokens.concat(new_token);

        // for 2+ inference, we don't need encoder hidden states as input to OV model
        delete decoder_input.encoder_hidden_states;

        // -----------------------------------DECODER 2 INFERENCE-----------------------------------------
        // prepare inputs for decoder kv cache

        // create 1x1 array for input_ids
        // decoder_input['input_ids'] = new this.ort.Tensor('int32', new Int32Array([new_token]), [1, 1]);
        this.context.writeBuffer(this.models['decoder_cached'].inputs.input_ids, new Int32Array([new_token]), 1);

        // pad attention mask to max_seq_length
        // decoder_input['attention_mask'] = new this.ort.Tensor('int64', attention_mask_update(this.ort, new BigInt64Array([1n, 1n, 1n, 1n]),
        //     0, this.max_sequence_length, this.num_init_tokens), [1, 128]);
        const initial_attention_mask = attention_mask_update(this.ort, new BigInt64Array([1n, 1n, 1n, 1n]), 0, this.max_sequence_length, this.num_init_tokens)
        this.context.writeBuffer(this.models['decoder_cached'].inputs.attention_mask, initial_attention_mask.buffer, initial_attention_mask.byteLength);

        // create position_ids as input, value should be same of No. of prefill tokens
        // decoder_input['position_ids'] = new this.ort.Tensor('int32', new Int32Array([this.num_init_tokens]), [1]);
        this.context.writeBuffer(this.models['decoder_cached'].inputs.position_ids, new Int32Array([this.num_init_tokens]), 1);

        // // modify the self attention kv cache in place
        this.dispatch_graph('pad_cache');

        const position_ids = new Int32Array(decoder_input['position_ids'].cpuData.buffer);
        // run complete inference for every item in dataset
        for (let i = 4; i < this.max_sequence_length; i++) {
            if (this.verbose) {
                console.log(`  decoder iteration ${i-3}: input preparation time: ${(performance.now() - start).toFixed(2)} ms`);
            }
            start = performance.now();
            // const decoder_cached_output = await this.models['decoder_cached']['sess'].run(decoder_input);
            this.context.dispatch(this.models['decoder_cached'].graph, this.models['decoder_cached'].inputs, this.models['decoder_cached'].outputs);
            if (this.verbose) {
                console.log(`  decoder iteration ${i-3}: inference time: ${(performance.now() - start).toFixed(2)} ms`);
            }
            start = performance.now();
            // find out the token with highest probability, cast INT64 to INT32
            // let logits = decoder_cached_output['logits']['cpuData'];
            const logitsBuffer = await this.context.readBuffer(this.models['decoder_cached'].outputs.logits);
            let logits = new Uint16Array(logitsBuffer);

            if (this.dataType == 'float16') {
                logits = convertToFloat32Array(logits);
            }
            const new_token = get_new_tokens(logits, [1, 1, 51865]);

            // add token to final buffer
            tokens = tokens.concat(new_token);
            // break if the new token is eos_token_id: 50257 (end of sequence)
            if (new_token == 50257) {
                break;
            }
            // ----------------------------------POST PROCESSING---------------------------------------
            // the following code creates decoder input for the next inference
            // decoder_input['input_ids'] = new this.ort.Tensor('int32', new Int32Array([new_token]), [1, 1]);
            this.context.writeBuffer(this.models['decoder_cached'].inputs.input_ids, new Int32Array([new_token]), 1);

            // increment the position_ids
            // position_ids[0] = position_ids[0] + 1;
            this.dispatch_increment_position_ids_graphs();

            // update mask using position id
            // attention_mask_update(this.ort, decoder_input['attention_mask'].cpuData, i, this.max_sequence_length, this.num_init_tokens, position_ids[0]);
            this.dispatch_update_attention_mask_graphs();

            // modify the kv cache in place
            // cache_update(this.ort, decoder_input, decoder_cached_output, i, this.max_sequence_length, this.num_init_tokens, position_ids[0], this.dataType);
            this.dispatch_update_cache_graphs();
        }

        // add token to sentence decode time
        const sentence = await this.tokenizer.decode(tokens, { skip_special_tokens: true });
        if (this.verbose) {
            console.log(`  post-processing time: ${(performance.now() - start).toFixed(2)} ms`);
        }
        return sentence;
    }
}
