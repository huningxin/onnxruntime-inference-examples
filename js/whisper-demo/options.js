// Import our custom CSS
import './scss/styles.scss'

async function updateUi() {
  // Access settings from storage with default values.
  const {
    deviceType,
    chunkLength,
    maxChunkLength,
    accumulateSubChunks,
    maxAudioLength } = await chrome.storage.local.get({
    dataType: 'float16',
    deviceType: 'gpu',
    chunkLength: '0.08',
    maxChunkLength: 2,
    accumulateSubChunks: false,
    maxAudioLength: 10
  });

  // Update UI with current values.
  document.getElementById("deviceType").value = deviceType;
  document.getElementById("chunkLength").value = chunkLength;
  document.getElementById("maxChunkLength").value = maxChunkLength;
  document.getElementById("accumulateSubChunks").checked = accumulateSubChunks;
  document.getElementById("maxAudioLength").value = maxAudioLength;
}

async function onSave() {
  const deviceType = document.getElementById("deviceType").value;
  const chunkLength = document.getElementById("chunkLength").value;
  const maxChunkLength = document.getElementById("maxChunkLength").value;
  const accumulateSubChunks = document.getElementById("accumulateSubChunks").checked;
  const maxAudioLength = document.getElementById("maxAudioLength").value;

  const warning = document.getElementById("warning")

  if (parseFloat(maxChunkLength) < parseFloat(chunkLength)) {
    warning.innerHTML = "Interim audio length should be larger than VAD audio chunk length.";
    warning.style.display = "block";
    return;
  }

  if (parseFloat(maxAudioLength) < parseFloat(maxChunkLength)) {
    warning.innerHTML = "Final audio length should be larger than interim audio length.";
    warning.style.display = "block";
    return;
  }

  warning.innerHTML = "";
  warning.style.display = "none";

  // Save to storage.
  chrome.storage.local.set({
    deviceType,
    chunkLength,
    maxChunkLength,
    accumulateSubChunks,
    maxAudioLength
  });
}

// Update UI immediately, and on any storage changes.
updateUi();
chrome.storage.local.onChanged.addListener(updateUi);

// Register listener for save button click.
document.getElementById('save').addEventListener('click', onSave);
