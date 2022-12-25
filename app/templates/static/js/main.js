function getModelName() {

    var e = document.getElementById("model_name_dropdown");
    var value = e.options[e.selectedIndex].value;

    var e = document.getElementById("model_name");
    e.setAttribute('value', value);
}

function selectVideo(self) {
    var file = self.files[0];
    var reader = new FileReader();

    reader.onload = function(e) {
        var src = e.target.result;
        let targetVideoFormat = 'mp4';
        alert('start convert');
        let convertedVideoDataObj = convert(src, targetVideoFormat);
        alert("converted");
        var video = document.getElementById('video');
        var source = document.getElementById('source');

        source.setAttribute('src', convertedVideoDataObj);
        video.load();
        video.play();
    };

    reader.readAsDataURL(file);
    // alert("Read");
}

function printValue(){
    var name = document.getElementById("model_name").getAttribute('value');
    alert(name)
    var vid = document.getElementById("model_name").files;
    alert(name)

}