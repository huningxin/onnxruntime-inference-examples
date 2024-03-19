// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run whisper in onnxruntime-web.
//

import {Whisper} from './whisper.js';
import {log} from './utils.js';

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

// stats
let total_processing_time = 0;
let total_processing_count = 0;

// some dom shortcuts
let record;
let transcribe;
let progress;
let audio_src;

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
    transcribe.disabled = false;
    progress.style.width = "0%";
    progress.parentNode.style.display = "none";
}

// called when document is loaded
document.addEventListener("DOMContentLoaded", async () => {
    audio_src = document.querySelector('audio');
    record = document.getElementById('record');
    transcribe = document.getElementById('transcribe');
    progress = document.getElementById('progress');
    transcribe.disabled = true;
    progress.parentNode.style.display = "none";
    updateConfig();

    // click on Record
    record.addEventListener("click", (e) => {
        if (e.currentTarget.innerText == "Record") {
            e.currentTarget.innerText = "Stop Recording";
            startRecord(0);
        }
        else {
            e.currentTarget.innerText = "Record";
            stopRecord();
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
    log("loading model");
    try {
        whisper = new Whisper('https://huggingface.co/lwanming/whisper-base-static-shape/resolve/main/', provider, dataType);
        // whisper = new Whisper('./models/', provider, dataType);
        await whisper.create_whisper_processor();
        await whisper.create_whisper_tokenizer();
        await whisper.create_ort_sessions();
        ready();
        context = new AudioContext({
            sampleRate: kSampleRate,
            channelCount: 1,
            echoCancellation: false,
            autoGainControl: true,
            noiseSuppression: true,
        });
        if (!context) {
            throw new Error("no AudioContext, make sure domain has access to Microphone");
        }
    } catch (e) {
        log(`Error: ${e}`);
    }
});

// report progress
function update_status(t) {
    total_processing_time += t;
    total_processing_count += 1;
    const avg = 1000 * 30 * total_processing_count / total_processing_time;
    document.getElementById('latency').innerText = `${avg.toFixed(1)} x realtime`;
}

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
            const start = performance.now();
            const ret = await whisper.run(xa, kSampleRate);
            const diff = performance.now() - start;
            update_status(diff);

            // append results to textarea 
            const textarea = document.getElementById('outputText');
            textarea.value += ret; // `${ret.str.data[0]}\n`;
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
        log(`${document.getElementById('latency').innerText}, total ${processing_time.toFixed(1)}sec for ${total.toFixed(1)}sec`);
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
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
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
