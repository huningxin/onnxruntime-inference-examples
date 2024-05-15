import { AutoProcessor, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.15.1';

import { initWhisper, process_audio, startSpeech, stopSpeech, getAudioContext } from './main.js';

import { log } from './utils.js';

// some dom shortcuts
let record;
let speech;
let transcribe;
let progress;
let audio_src;
let textarea;

let mediaRecorder;
let stream;
const kIntervalAudio_ms = 1000;

let speechToText = '';

const options = {};

function updateConfig() {
    const query = window.location.search.substring('1');
    const providers = ['webnn', 'webgpu', 'wasm'];
    const deviceTypes = ['cpu', 'gpu', 'npu']
    const dataTypes = ['float32', 'float16'];
    let vars = query.split('&');
    for (let i = 0; i < vars.length; i++) {
        let pair = vars[i].split('=');
        if (pair[0] == 'provider' && providers.includes(pair[1])) {
            options.provider = pair[1];
        }
        if (pair[0] == 'deviceType' && deviceTypes.includes(pair[1])) {
            options.deviceType = pair[1];
        }
        if (pair[0] == 'dataType' && dataTypes.includes(pair[1])) {
            options.dataType = pair[1];
        }
        if (pair[0] == 'maxChunkLength') {
            options.maxChunkLength = parseFloat(pair[1]);
        }
        if (pair[0] == 'chunkLength') {
            options.chunkLength = parseFloat(pair[1]);
        }
        if (pair[0] == 'maxAudioLength') {
            options.maxAudioLength = parseFloat(pair[1]);
        }
        if (pair[0] == 'verbose') {
            options.verbose = pair[1].toLowerCase() === 'true';
        }
        if (pair[0] == 'accumulateSubChunks') {
            options.accumulateSubChunks = pair[1].toLowerCase() === 'true';
        }
        if (pair[0] == 'mask_4d') {
            options.mask_4d = pair[1].toLowerCase() === 'true';
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
    log('whisper ready');
}

const recognitionClient = {
    _onstart: () => {
        log('Recognition starts');
    },
    _onend: () => {
        log('Recognition ends');
    },
    _onresult: (transcript, isFinal) => {
        console.log(`${isFinal ? 'final' : 'interim'} result:${transcript}`);
        if (!isFinal) {
            textarea.value = speechToText + transcript;
        } else {
            speechToText += transcript;
            textarea.value = speechToText;
        }
        textarea.scrollTop = textarea.scrollHeight;
    },
    _onerror: (e) => {
        log(`Recognition error: ${e}`);
    }
};

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
            // If audio has a source, do speech recognition from it otherwise from a mic.
            if (await startSpeech(recognitionClient, audio_src.readyState ? audio_src : undefined)) {
                speech.innerText = "Stop Speech";
            }
            if (audio_src.readyState) {
                audio_src.play();
            }
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
        if (files.length > 0) {
            audio_src.src = URL.createObjectURL(files[0]);
        } else {
            audio_src.src = '';
        }
    }

    if (await initWhisper(ort, AutoProcessor, AutoTokenizer, options)) {
        ready();
    }
});

// transcribe audio source
async function transcribe_file() {
    if (!audio_src.readyState) {
        log("Error: set some Audio input");
        return;
    }

    busy();
    log("start transcribing...");
    try {
        const buffer = await (await fetch(audio_src.src)).arrayBuffer();
        const audioBuffer = await getAudioContext().decodeAudioData(buffer);
        var offlineContext = new OfflineAudioContext(audioBuffer.numberOfChannels, audioBuffer.length, audioBuffer.sampleRate);
        var source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start();
        const renderedBuffer = await offlineContext.startRendering();
        const audio = renderedBuffer.getChannelData(0);
        await process_audio(audio, performance.now(), 0, 0, textarea, progress);
        ready();
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
                        echoCancellation: true,
                        autoGainControl: true,
                        noiseSuppression: true,
                        channelCount: 1,
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
