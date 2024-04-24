function loadScript(scriptName, options) {
  var s = document.createElement('script');
  s.src = chrome.runtime.getURL(scriptName);
  s.dataset.params = JSON.stringify(options);
  console.log(`Loads ${scriptName}`);
  s.onload = function() { this.remove(); };
  (document.head || document.documentElement).appendChild(s);
}
const options = await chrome.storage.local.get({
  provider: 'webnn',
  dataType: 'float16',
  deviceType: 'gpu',
  chunkLength: '0.1',
  maxChunkLength: 2,
  accumulateSubChunks: false,
  maxAudioLength: 10
});
options.extensionId = chrome.runtime.id;
loadScript('polyfill.js', options);