// import * as tfnode from '@tensorflow/tfjs-node'
console.log("on temo.js");
var loadFile = function (event) {
    var image = document.getElementById("image");
    image.src = URL.createObjectURL(event.target.files[0]);
    document.getElementById("image").style.display="block";
     };
    
    // const classifier = ml5.imageClassifier
    //  ("MobileNet", modelLoaded);

    //  // When the model is loaded
    //  function modelLoaded() {
    //  console.log("Model Loaded!");
    //  }
    tf.loadLayersModel("model/model.json").then(function(model) {
    window.model = model;
    console.log("my model loaded");
    });
    
    // var canvas = document.createElement('canvas');
    // var context = canvas.getContext('2d');
    // var img = document.getElementById('image');
    // canvas.width = img.width;
    // canvas.height = img.height;
    // context.drawImage(img, 0, 0 );
    // var myData = context.getImageData(0, 0, img.width, img.height);
    // console.log(myData);
    function processImage(path) {

        const imageSize = 224
        const imageBuffer =  fs.readFileSync(path); // can also use the async readFile instead
        // get tensor out of the buffer
        image = tfnode.node.decodeImage(imageBuffer, 3);
        // dtype to float
        image = image.cast('float32').div(255);
        // resize the image
        image = tf.image.resizeBilinear(image, size = [imageSize, imageSize]); // can also use tf.image.resizeNearestNeighbor
        image = image.expandDims(); // to add the most left axis of size 1
        // return image.shape
        console.log(image);
    }
     function predict() {
        
        var canvas = document.createElement('canvas');
        var context = canvas.getContext('2d');
        var img = document.getElementById('image');
        processImage(img.src);
        canvas.width = 224;
        canvas.height = 224;
        context.drawImage(img, 0, 0 );
        var myData = context.getImageData(0, 0, 224, 224).data;
        console.log(myData);
        console.log(myData.length);
        var normalArray = Array.from(myData);
        console.log(normalArray.length);
        var input = [];
        for(var i = 0; i < myData.length; i += 1) {
        input.push(myData[i] / 255);
        }
        window.model.predict([tf.tensor(input).reshape([null, 224, 224, 3])]).array().then(function(scores){
        scores = scores[0];
        predicted = scores.indexOf(Math.max(...scores));
        // $('#number').html(predicted);
        console.log(predicted);
        });
    //  classifier.predict(document.getElementById("image"), 
    // 		  function (err, results) {
    // 			//   alert(results[0].label);
    // 			alert(results[0].label);
    // 		  });
    }