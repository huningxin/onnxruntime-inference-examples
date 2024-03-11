import { AutoProcessor, AutoTokenizer, read_audio } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.15.1';

//'@xenova/transformers';
import { get_new_tokens } from './generation_utils.js';
import { attention_mask_update, cache_update } from './post_processing.js';

export function log(i) { 
    console.log(i);
    document.getElementById('status').innerText += `\n[${performance.now().toFixed(2)}] ` + i;
}

// wrapper around onnxruntime and model
export class Whisper {
    constructor(url, provider) {
        this.url = url;
        this.provider = provider;
        ort.env.wasm.simd = true;

        this.models = {
            'encoder': { url: 'whisper_base_encoder.onnx', sess: null },
            'decoder': { url: 'whisper_base_decoder_static_non_kvcache_logits.onnx', sess: null },
            'decoder_cached': { url: 'whisper_base_decoder_static_kvcache_logits.onnx', sess: null },
        };

        this.max_sequence_length = 128;
        // No. of tokens to be used for decoder 1st inference
        this.num_init_tokens = 4;
        // Whisper was trained using 16000 Hz as sampling rate, fixing this value for dataset preparation
        this.sampling_rate = 16000;
        // set precision of different inputs
        this.cache_precision = Float32Array;
        this.processor = null;
        this.tokenizer = null;
    }

async create_whisper_processor() {
    // processor contains feature extractor
    this.processor = await AutoProcessor.from_pretrained('openai/whisper-tiny.en');
}

async create_whisper_tokenizer() {
    // processor contains feature extractor
    this.tokenizer = await AutoTokenizer.from_pretrained('openai/whisper-tiny.en', { config: { do_normalize: true } });
}

async create_ort_sessions() {
    const options = {
        executionProviders: [{
            name: this.provider,
            deviceType: 'gpu',
        }],
    };
    // options.logSeverityLevel = 0;
    for (let name of Object.keys(this.models)) {
        try {
            this.models[name]['sess'] = await ort.InferenceSession.create(this.url + this.models[name]['url'], options);
            log(`Model ${this.models[name]['url']} loaded`);
        } catch (e) {
            log(`Error: ${e}`);
        }
    };
}

async run(audio_data, sampling_rate) {
    // -----------------------------------FEATURE EXTRACTION-----------------------------------------
    // const audio = await read_audio('https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac', 16000);
    // const audio = await read_audio(audio_data, sampling_rate);
    const { input_features } = await this.processor(audio_data);

    // -----------------------------------ENCODER INFERENCE-----------------------------------------
    // run encoder to get output
    const encoder_hidden_states = await this.models['encoder']['sess'].run({
        input_features: new ort.Tensor(input_features.type, input_features.data, input_features.dims)
    });

    // -----------------------------------DECODER 1ST INFERENCE-----------------------------------------
    // create list of tokens for english language and transcribe task, no need of time stamps
    // TODO: CHANGE FROM HARDCODED VALUES
    let tokens = [50258, 50259, 50359, 50363];
    const attention_mask = [1, 1, 1, 1];
    // create decoder input for the first inference
    const decoder_input = {
        'input_ids': new ort.Tensor('int32', new Int32Array(tokens), [1, 4]),
        'attention_mask': new ort.Tensor('int32', new Int32Array(attention_mask), [1, 4]),
        'encoder_hidden_states': encoder_hidden_states['last_hidden_state'],
    };
    // run the first inference which generates SA and CA KV cache
    const decoder_output = await this.models['decoder']['sess'].run(decoder_input);

    const logits = decoder_output['logits']['cpuData'];
    // find out the token with highest probability, cast INT64 to INT32
    const new_token = get_new_tokens(logits, [1, 4, 51865]);

    // add token to final buffer
    tokens = tokens.concat(new_token);

    // for 2+ inference, we don't need encoder hidden states as input to OV model
    delete decoder_input.encoder_hidden_states;

    // -----------------------------------DECODER 2 INFERENCE-----------------------------------------
    // prepare inputs for decoder kv cache

    // create 1x1 array for input_ids
    decoder_input['input_ids'] = new ort.Tensor('int32', new Int32Array([new_token]), [1, 1]);

    // pad attention mask to max_seq_length
    decoder_input['attention_mask'] = new ort.Tensor('int64', attention_mask_update(new BigInt64Array([1n, 1n, 1n, 1n]),
        0, this.max_sequence_length, this.num_init_tokens), [1, 128]);
    // create past_key_values_length as input, value should be same of No. of prefill tokens
    decoder_input['past_key_values_length'] = new ort.Tensor('int32', new Int32Array([this.num_init_tokens]), [1]);

    // fill decoder kv cache model inputs with cross attention KV cache data from decoder 1st inference
    for (let i = 0; i < 6; i++) {
        decoder_input[`past_key_values.${i}.encoder.key`] = decoder_output[`present_key_values.${i}.encoder.key`];
        decoder_input[`past_key_values.${i}.encoder.value`] = decoder_output[`present_key_values.${i}.encoder.value`];
    }

    // modify the self attention kv cache in place
    cache_update(
        decoder_input,
        decoder_output,
        0,
        this.max_sequence_length,
        this.num_init_tokens,
        this.cache_precision);
    const past_key_values_length = new Int32Array(decoder_input['past_key_values_length'].cpuData.buffer);
    // run complete inference for every item in dataset
    for (let i = 4; i < this.max_sequence_length; i++) {
        const decoder_cached_output = await this.models['decoder_cached']['sess'].run(decoder_input);

        // find out the token with highest probability, cast INT64 to INT32
        const new_token = get_new_tokens(decoder_cached_output['logits']['cpuData'], [1, 1, 51865]);

        // add token to final buffer
        tokens = tokens.concat(new_token);

        // ----------------------------------POST PROCESSING---------------------------------------
        // the following code creates decoder input for the next inference
        decoder_input['input_ids'] = new ort.Tensor('int32', new Int32Array([new_token]), [1, 1]);

        // increment the past_key_values_length
        past_key_values_length[0] = past_key_values_length[0] + 1;

        // update mask using position id
        attention_mask_update(decoder_input['attention_mask'].cpuData, i, this.max_sequence_length, this.num_init_tokens, past_key_values_length[0]);
        // modify the kv cache in place
        cache_update(decoder_input, decoder_cached_output, i, past_key_values_length[0]);
        // break if the new token is eos_token_id: 50256 (end of sequence)
        if (new_token == 50256) {
            break;
        }
    }

    // add token to sentence decode time
    const sentence = await this.tokenizer.decode(tokens, { skip_special_tokens: true });

    // TODO
    // norm_ref = self.processor.tokenizer._normalize(input_data['reference']);
    // norm_ov = self.processor.tokenizer._normalize(sentence);
    console.log('tokens: ', tokens);
    console.log('result: ', sentence);
    return sentence;
}
}

async function main() {
    const whisper = new Whisper('./models/');
    await whisper.create_whisper_processor();
    await whisper.create_whisper_tokenizer();
    await whisper.create_ort_sessions();
    await whisper.run();
}

// main();