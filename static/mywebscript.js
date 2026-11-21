/* Send the text typed by the user to the Flask backend. */
function RunSentimentAnalysis() {
    var textToAnalyze = document.getElementById('textToAnalyze').value;
    var responseBox = document.getElementById('system_response');

    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/emotionDetector?textToAnalyze=' + encodeURIComponent(textToAnalyze), true);

    xhr.onreadystatechange = function () {
        if (xhr.readyState !== 4) {
            return;
        }
        if (xhr.status === 200) {
            responseBox.className = 'result';
            responseBox.innerHTML = xhr.responseText;
        } else {
            responseBox.className = 'error';
            responseBox.textContent = xhr.responseText || 'Invalid text! Please try again!';
        }
    };

    xhr.send();
}
