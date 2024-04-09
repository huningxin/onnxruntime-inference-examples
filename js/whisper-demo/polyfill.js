import { setRecognition, startSpeech, stopSpeech, initAudio, initWhisper} from './main.js';

const extensionId = JSON.parse(document.currentScript.dataset.params).extensionId;

class WhisperSpeechRecognition {
    constructor() {
        console.log('WhisperSpeechRecognition.constructor');
        this.ready_ = false;
        this.continuous = false;
        this.interimResults = false;
        setRecognition(this, extensionId);
        initAudio();
        initWhisper();
    }

    _onready() {
        this.ready_ = true;
    }

    async start() {
        console.log('WhisperSpeechRecognition.start');
        if (!this.ready_) {
            console.warn('WhisperSpeechRecognition is not ready');
            return;
        }
        startSpeech();
    }

    async stop() {
        console.log('WhisperSpeechRecognition.stop');
        if (!this.ready_) {
            console.warn('WhisperSpeechRecognition is not ready');
            return;
        }
        stopSpeech();
    }

    async abort() {
        console.log('WhisperSpeechRecognition.abort');
        if (!this.ready_) {
            console.warn('WhisperSpeechRecognition is not ready');
            return;
        }
        stopSpeech();
    }

    _onstart() {
        console.log('WhisperSpeechRecognition._onstart');
        this.onstart();
    }

    _onend() {
        console.log('WhisperSpeechRecognition._onend');
        this.onend();
    }

    _onresult(transcript, isFinal) {
        console.log(`WhisperSpeechRecognition._onresult: transcript: ${transcript}, isFinal: ${isFinal}`);
        const results = [
            {
                0: {
                transcript,
                confidence: 1,
                },
                isFinal: isFinal,
            },];
        this.onresult({ results, resultIndex: 0 })
    }

    _onerror(e) {
        console.log('WhisperSpeechRecognition._onresult');
        this.onerror(e);
    }

    onstart() {
        console.log('WhisperSpeechRecognition.onstart');
    }

    onend() {
        console.log('WhisperSpeechRecognition.onend');
    }

    onresult(e) {
        console.log('WhisperSpeechRecognition.onresult');
    }

    onerror(e) {
        console.log('WhisperSpeechRecognition.onresult');
    }
};

window.webkitSpeechRecognition = WhisperSpeechRecognition;
