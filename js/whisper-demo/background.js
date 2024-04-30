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
        const config = await chrome.storage.local.get();
        port.postMessage(config);
    });
});

const keepAlive = () => setInterval(chrome.runtime.getPlatformInfo, 20e3);
chrome.runtime.onStartup.addListener(keepAlive);
keepAlive();