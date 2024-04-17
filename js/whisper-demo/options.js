async function updateUi() {
  // Access settings from storage with default values.
  const { provider, dataType, maxChunkLength } = await chrome.storage.local.get({
    provider: 'webnn',
    dataType: 'float16',
    maxChunkLength: 1
  });

  // Update UI with current values.
  document.getElementById("provider").value = provider;
  document.getElementById("dataType").value = dataType;
  document.getElementById("maxChunkLength").value = maxChunkLength;
}

async function onSave() {
  const provider = document.getElementById("provider").value;
  const dataType = document.getElementById("dataType").value;
  const maxChunkLength = document.getElementById("maxChunkLength").value;

  // Save to storage.
  chrome.storage.local.set({
    provider,
    dataType,
    maxChunkLength
  });
}

// Update UI immediately, and on any storage changes.
updateUi();
chrome.storage.local.onChanged.addListener(updateUi);

// Register listener for save button click.
document.getElementById('save').addEventListener('click', onSave);
