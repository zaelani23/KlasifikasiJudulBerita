// ----Kolom deklarasi variabel-----
let input = document.querySelector('input');
let button = document.querySelector('button');
button.addEventListener('click', onClick);

const currentDate = new Date();

let isModelLoaded = false;
let model;
let word2index;

// Parameter data preprocessing
const maxlen = 14;
const vocab_size = 4100;
const padding = 'pre';
const truncating = 'pre';

var myVar;
// -----------------------------------

function myFunction() {
    myVar = setTimeout(showPage, 3000);
}

function showPage() {
    document.getElementById("year").innerHTML = currentDate.getFullYear();
    document.getElementById("loaderlabel").style.display = "none";
    document.getElementById("loader").style.display = "none";       
    document.getElementById("mainAPP").style.display = "block";
    document.getElementById('input').focus();
}

function detectWebGLContext () {
    // Create canvas element. The canvas is not added to the
    // document itself, so it is never displayed in the
    // browser window.
    var canvas = document.createElement("canvas");
    // Get WebGLRenderingContext from canvas element.
    var gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    // Report the result.
    if (gl && gl instanceof WebGLRenderingContext) {
        console.log("Congratulations! Your browser supports WebGL.");
        init();
    } else {
        // alert("Failed to get WebGL context. Your browser or device may not support WebGL.");
        Swal.fire(
          'Error!',
          'Failed to get WebGL context. Your browser or device may not support WebGL.',
          'error'
        );
    }
}

detectWebGLContext();

// ----Kolom fungsi `getInput()`-----
// Fungsi untuk mengambil input review
function getInput(){
    const newsTitle = document.getElementById('input')
    return newsTitle.value;
}
// -----------------------------------

// ----Kolom fungsi `padSequence()`-----
// Fungsi untuk melakukan padding
function padSequence(sequences, maxLen, padding='post', truncating = "post", pad_value = 0){
    return sequences.map(seq => {
        if (seq.length > maxLen) { //truncat
            if (truncating === 'pre'){
                seq.splice(0, seq.length - maxLen);
            } else {
                seq.splice(maxLen, seq.length - maxLen);
            }
        }
                
        if (seq.length < maxLen) {
            const pad = [];
            for (let i = 0; i < maxLen - seq.length; i++){
                pad.push(pad_value);
            }
            if (padding === 'pre') {
                seq = pad.concat(seq);
            } else {
                seq = seq.concat(pad);
            }
        }               
        return seq;
        });
}
// -----------------------------------


// ----Kolom fungsi `predict()`-----
// Fungsi untuk melakukan prediksi
function predict(inputText){

    // Mengubah input review ke dalam bentuk token
    const sequence = inputText.map(word => {
        let indexed = word2index[word];

        if (indexed === undefined){
            return 1; //change to oov value
        }
        return indexed;
    });
    
    // Melakukan padding
    const paddedSequence = padSequence([sequence], maxlen);
    const labelPrediksi = ['Bisnis', 'Health', 'Tekno'];
    const prediction = tf.tidy(() => {
        const input = tf.tensor2d(paddedSequence, [1, maxlen]);
        const result = model.predict(input);
        return labelPrediksi[result.dataSync().indexOf(Math.max(...result.dataSync()))];
    });

    return prediction;  

}
// -----------------------------------


// ----Kolom fungsi `onClick()`-----
// Fungsi yang dijalankan ketika tombol diclick
function onClick(){
    
    if(!isModelLoaded) {
        // alert('Model not loaded yet');
        Swal.fire(
          'Alert!',
          'Model not loaded yet!',
          'warning'
        );
        return;
    }

    if (getInput() === '') {
        // alert("Judul berita harus diisi!");
        Swal.fire(
          'Required!',
          'Judul berita harus diisi!',
          'error'
        );
        document.getElementById('input').focus();
        return;
    }
    
    try {
        const inputText = getInput().trim().toLowerCase().split(" ");

        // Lakukan prediksi
        let predictedLabel = predict(inputText); 

        // alert(predictedLabel);
        Swal.fire(
          predictedLabel,
          `Judul '${getInput()}' masuk kategori ${predictedLabel}`,
          'info'
        );

        document.getElementById('input').value = "";
    } catch (e) {
        Swal.fire(
          'Error!',
          `${e.message}`,
          'error'
        );
    }
}
// -----------------------------------


// ----Kolom fungsi `init()`-----
async function init(){

    // Memanggil model tfjs
    // model = await tf.loadLayersModel('http://127.0.0.1:5500/tfjs_model/model.json'); // Untuk VS Code Live Server
    model = await tf.loadLayersModel('http://127.0.0.1:8887/tfjs_model/model.json');
    isModelLoaded = true;

    //Memanggil word_index
    // const word_indexjson = await fetch('http://127.0.0.1:5500/word_index.json'); // Untuk VS Code Live Server
    const word_indexjson = await fetch('http://127.0.0.1:8887/word_index.json'); 
    word2index = await word_indexjson.json();
}
// -----------------------------------