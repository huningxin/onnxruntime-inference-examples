async function updateUi() {
  // Access settings from storage with default values.
  const {
    provider,
    dataType,
    deviceType,
    chunkLength,
    maxChunkLength,
    accumulateSubChunks,
    maxAudioLength } = await chrome.storage.local.get({
    provider: 'webnn',
    dataType: 'float16',
    deviceType: 'gpu',
    chunkLength: '0.1',
    maxChunkLength: 2,
    accumulateSubChunks: false,
    maxAudioLength: 10
  });

  // Update UI with current values.
  document.getElementById("provider").value = provider;
  document.getElementById("dataType").value = dataType;
  document.getElementById("deviceType").value = deviceType;
  document.getElementById("chunkLength").value = chunkLength;
  document.getElementById("maxChunkLength").value = maxChunkLength;
  document.getElementById("accumulateSubChunks").checked = accumulateSubChunks;
  document.getElementById("maxAudioLength").value = maxAudioLength;
}

async function onSave() {
  const provider = document.getElementById("provider").value;
  const dataType = document.getElementById("dataType").value;
  const deviceType = document.getElementById("deviceType").value;
  const chunkLength = document.getElementById("chunkLength").value;
  const maxChunkLength = document.getElementById("maxChunkLength").value;
  const accumulateSubChunks = document.getElementById("accumulateSubChunks").checked;
  const maxAudioLength = document.getElementById("maxAudioLength").value;

  // Save to storage.
  chrome.storage.local.set({
    provider,
    dataType,
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
