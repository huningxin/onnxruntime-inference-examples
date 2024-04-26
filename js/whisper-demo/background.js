chrome.runtime.onInstalled.addListener(({ reason }) => {
    if (reason == chrome.runtime.OnInstalledReason.INSTALL) {
        chrome.runtime.openOptionsPage();
    }
});
  
chrome.action.onClicked.addListener(() => {
    chrome.runtime.openOptionsPage();
});

chrome.runtime.onConnectExternal.addListener(function(port) {
    console.log('onConnectExternal');
    chrome.storage.local.onChanged.addListener(async () => {
        const config = await chrome.storage.local.get({
            provider: 'webnn',
            dataType: 'float16',
            deviceType: 'gpu',
            chunkLength: '0.08',
            maxChunkLength: 2,
            accumulateSubChunks: false,
            maxAudioLength: 10
          });
        port.postMessage(config);
    });
});