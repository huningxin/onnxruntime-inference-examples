// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run whisper in onnxruntime-web.
//

import { Whisper } from './whisper.js';
import { log } from './utils.js';

const kSampleRate = 16000;
const kIntervalAudio_ms = 1000;
const kSteps = kSampleRate * 30;
const kDelay = 100;

// whisper class
let whisper;

let provider = 'webnn';
let dataType = 'float32';

// audio context
var context = null;
let mediaRecorder;
let stream;

// some dom shortcuts
let record;
let speech;
let transcribe;
let progress;
let audio_src;
let textarea;

// for audio capture
// This enum states the current speech state.
const SpeechStates = {
    UNINITIALIZED: 0,
    PROCESSING: 1,
    PAUSED: 2,
    FINISHED: 3,
};
let speechState = SpeechStates.UNINITIALIZED;

let streamingNode = null;
let sourceNode = null;
let audioChunks = [];
let audioChunksIndex = 0;
let concatenatedAudio = null;
let chunkLength = 2; // audio chunk length in sec
let maxChunkLength = 10; // max audio length for an audio processing
// check if last transcription is completed, to avoid race condition
let isLastTransCompleted = false;
let isProcessing = false;
let speechToText = '';

const blacklistTags = [
    '[inaudible]',
    '[INAUDIBLE]',
    '[BLANK_AUDIO]',
    ' [inaudible]',
    ' [INAUDIBLE]',
    ' [BLANK_AUDIO]',
    '[ Inaudible ]',
    ' [no audio]',
    '[no audio]',
    '[silent]',
];

function updateConfig() {
    const query = window.location.search.substring('1');
    const providers = ['webnn', 'webgpu', 'wasm'];
    const dataTypes = ['float32', 'float16'];
    let vars = query.split('&');
    for (let i = 0; i < vars.length; i++) {
        let pair = vars[i].split('=');
        if (pair[0] == 'provider' && providers.includes(pair[1])) {
            provider = pair[1];
        }
        if (pair[0] == 'dataType' && dataTypes.includes(pair[1])) {
            dataType = pair[1];
        }
        if (pair[0] == 'chunkLength') {
            chunkLength = parseInt(pair[1]);
        }
        if (pair[0] == 'maxChunkLength') {
            maxChunkLength = parseInt(pair[1]);
        }
    }
}

// transcribe active
function busy() {
    transcribe.disabled = true;
    progress.parentNode.style.display = "block";
    document.getElementById("outputText").value = "";
    document.getElementById('latency').innerText = "";
}

// transcribe done
function ready() {
    speech.disabled = false;
    transcribe.disabled = false;
    progress.style.width = "0%";
    progress.parentNode.style.display = "none";
}

// called when document is loaded
document.addEventListener("DOMContentLoaded", async () => {
    audio_src = document.querySelector('audio');
    record = document.getElementById('record');
    speech = document.getElementById('speech');
    transcribe = document.getElementById('transcribe');
    progress = document.getElementById('progress');
    textarea = document.getElementById('outputText');
    transcribe.disabled = true;
    speech.disabled = true;
    progress.parentNode.style.display = "none";
    updateConfig();

    // click on Record
    record.addEventListener("click", (e) => {
        if (e.currentTarget.innerText == "Record") {
            e.currentTarget.innerText = "Stop Recording";
            startRecord();
        }
        else {
            e.currentTarget.innerText = "Record";
            stopRecord();
        }
    });

    // click on Speech
    speech.addEventListener("click", async (e) => {
        if (e.currentTarget.innerText == "Start Speech") {
            if (isProcessing) {
                log('Last speech-to-text has not completed yet, try later...');
                return;
            }
            e.currentTarget.innerText = "Stop Speech";
            await startSpeech();
        }
        else {
            e.currentTarget.innerText = "Start Speech";
            await stopSpeech();
        }
    });

    // click on Transcribe
    transcribe.addEventListener("click", () => {
        transcribe_file();
    });

    // drop file
    document.getElementById("file-upload").onchange = function (evt) {
        let target = evt.target || window.event.src, files = target.files;
        audio_src.src = URL.createObjectURL(files[0]);
    }
    log(`Execution provider: ${provider}`);
    log("loading model...");
    try {
        context = new AudioContext({
            sampleRate: kSampleRate,
            channelCount: 1,
            echoCancellation: false,
            autoGainControl: true,
            noiseSuppression: true,
        });
        const whisper_url = location.href.includes('github.io') ?
            'https://huggingface.co/lwanming/whisper-base-static-shape/resolve/main/' :
            './models/';
        whisper = new Whisper(whisper_url, provider, dataType);
        await whisper.create_whisper_processor();
        await whisper.create_whisper_tokenizer();
        await whisper.create_ort_sessions();
        log("Ready to transcribe...");
        ready();
        if (!context) {
            throw new Error("no AudioContext, make sure domain has access to Microphone");
        }
    } catch (e) {
        log(`Error: ${e}`);
    }
});

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// process audio buffer
async function process_audio(audio, starttime, idx, pos) {
    if (idx < audio.length) {
        // not done
        try {
            // update progress bar
            progress.style.width = (idx * 100 / audio.length).toFixed(1) + "%";
            progress.textContent = progress.style.width;
            await sleep(kDelay);
            // run inference for 30 sec
            const xa = audio.slice(idx, idx + kSteps);
            const ret = await whisper.run(xa, kSampleRate);
            // append results to textarea 
            textarea.value += ret;
            textarea.scrollTop = textarea.scrollHeight;
            await sleep(kDelay);
            process_audio(audio, starttime, idx + kSteps, pos + 30);
        } catch (e) {
            log(`Error: ${e}`);
            ready();
        }
    } else {
        // done with audio buffer
        const processing_time = ((performance.now() - starttime) / 1000);
        const total = (audio.length / kSampleRate);
        document.getElementById('latency').innerText = `${(total / processing_time).toFixed(1)} x realtime`;
        log(`${document.getElementById('latency').innerText}, total ${processing_time.toFixed(1)}sec processing time for ${total.toFixed(1)}sec audio`);
        ready();
    }
}

// transcribe audio source
async function transcribe_file() {
    if (audio_src.src == "") {
        log("Error: set some Audio input");
        return;
    }

    busy();
    log("start transcribe ...");
    try {
        const buffer = await (await fetch(audio_src.src)).arrayBuffer();
        const audioBuffer = await context.decodeAudioData(buffer);
        var offlineContext = new OfflineAudioContext(audioBuffer.numberOfChannels, audioBuffer.length, audioBuffer.sampleRate);
        var source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start();
        const renderedBuffer = await offlineContext.startRendering();
        const audio = renderedBuffer.getChannelData(0);
        process_audio(audio, performance.now(), 0, 0);
    }
    catch (e) {
        log(`Error: ${e}`);
        ready();
    }
}

// start recording
async function startRecord() {
    if (mediaRecorder === undefined) {
        try {
            if (!stream) {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        autoGainControl: false,
                        noiseSuppression: false,
                        latency: 0
                    }
                });
            }
            mediaRecorder = new MediaRecorder(stream);
        } catch (e) {
            record.innerText = "Record";
            log(`Access to Microphone, ${e}`);
        }
    }
    let recording_start = performance.now();
    let chunks = [];

    mediaRecorder.ondataavailable = (e) => {
        chunks.push(e.data);
        document.getElementById('latency').innerText = `recorded: ${((performance.now() - recording_start) / 1000).toFixed(1)}sec`;
    }

    mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { 'type': 'audio/ogg; codecs=opus' });
        log(`recorded ${((performance.now() - recording_start) / 1000).toFixed(1)}sec audio`);
        audio_src.src = window.URL.createObjectURL(blob);
    };
    mediaRecorder.start(kIntervalAudio_ms);
}

// stop recording
function stopRecord() {
    if (mediaRecorder) {
        mediaRecorder.stop();
        mediaRecorder = undefined;
    }
}

// start speech
async function startSpeech() {
    await captureAudioStream();
    if (streamingNode != null) {
        streamingNode.port.postMessage({ message: "STOP_PROCESSING", data: false });
    }
}

// stop speech
async function stopSpeech() {
    if (streamingNode != null) {
        streamingNode.port.postMessage({ message: "STOP_PROCESSING", data: true });
        speechState = SpeechStates.PAUSED;
    }
    // if (stream) {
    //     stream.getTracks().forEach(track => track.stop());
    // }
    // if (context) {
    //     // context.close().then(() => context = null);
    //     await context.suspend();
    // }
}

// use AudioWorklet API to capture real-time audio
async function captureAudioStream() {
    try {
        if (context && context.state === 'suspended') {
            await context.resume();
        }
        // Get user's microphone and connect it to the AudioContext.
        if (!stream) {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    autoGainControl: false,
                    noiseSuppression: false,
                    latency: 0
                }
            });
        }
        if (streamingNode) {
            return;
        }
        // clear output context
        textarea.value = '';
        sourceNode = new MediaStreamAudioSourceNode(context, { mediaStream: stream });
        await context.audioWorklet.addModule('streaming-processor.js');
        const streamProperties = {
            numberOfChannels: 1,
            sampleRate: context.sampleRate,
            chunkLength: chunkLength,
        };
        streamingNode = new AudioWorkletNode(
            context,
            'streaming-processor',
            {
                processorOptions: streamProperties,
            },
        );

        streamingNode.port.onmessage = async (e) => {
            if (e.data.message === 'START_TRANSCRIBE') {
                audioChunks.push(e.data.buffer);
                // first time init the processAudioBuffer
                if (speechState == SpeechStates.UNINITIALIZED) {
                    speechState = SpeechStates.PROCESSING;
                    await processAudioBuffer();
                }
                // new audio is coming, and no audio is processing
                if (isLastTransCompleted && !isProcessing && (audioChunks.length > audioChunksIndex)) {
                    await processAudioBuffer();
                }
            }
        };

        sourceNode
            .connect(streamingNode)
            .connect(context.destination);
    } catch (e) {
        log(`Error on capturing audio: ${e}`);
    }
}

async function processAudioBuffer() {
    isLastTransCompleted = false;
    // meet maximum audio chunks, cut the last 30 sec audio, clear concatenatedAudio, audioChunksIndex
    if (audioChunksIndex == maxChunkLength / chunkLength) {
        audioChunks.splice(0, maxChunkLength / chunkLength);
        concatenatedAudio = null;
        audioChunksIndex = 0;
    }

    // pass the first audio chunk directly when audioChunksIndex is 0
    if (audioChunksIndex == 0) {
        concatenatedAudio = audioChunks[0];
    } else {
        // concat the previous audio chunks
        concatenatedAudio = concatAudio(concatenatedAudio, audioChunks[audioChunksIndex]);
    }

    const start = performance.now();
    const ret = await whisper.run(concatenatedAudio, kSampleRate);
    console.log(`${concatenatedAudio.length / kSampleRate} sec audio transcription time: ${((performance.now() - start) / 1000).toFixed(2)}sec`);
    // ignore slient, inaudible audio, i.e. '[BLANK_AUDIO]'
    // if (!blacklistTags.includes(ret)) {
    // append results to textarea
    if (audioChunksIndex == (maxChunkLength / chunkLength - 1)) {
        speechToText += ret; // append for each maxChunkLength/chunkLength iteration
        textarea.value = speechToText;
    } else {
        textarea.value = speechToText + ret; // refresh in a maxChunkLength/chunkLength iteration
    }
    textarea.scrollTop = textarea.scrollHeight;

    audioChunksIndex++;
    isLastTransCompleted = true;
    if (audioChunks.length > audioChunksIndex && audioChunks.length != 0) {
        // recusive audioBuffer in audioChunks
        isProcessing = true;
        await processAudioBuffer();
    } else {
        isProcessing = false;
    }
}

function concatAudio(lastAudio, newAudio) {
    let nextAudio = new Float32Array(lastAudio.length + newAudio.length);
    nextAudio.set(lastAudio);
    nextAudio.set(newAudio, lastAudio.length);
    return nextAudio;
}
