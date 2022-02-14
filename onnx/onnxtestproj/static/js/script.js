const sess = new onnx.InferenceSession();
const loadModelPromise = sess.loadModel('/static/js/simple_linear.onnx');

const pred = document.getElementById('pred');
// pred.innerHTML = 'test'

const testinput = [new Tensor(new Float32Array([9.0]), 'float32', [1,1]),];

const button = document.querySelector("#btn1");


loadModelPromise.then(() => {
    console.log('loaded model');
    // get the input from the textbox
    button.addEventListener('click', function(){
        var input = document.getElementById('inpnum');
        input = parseFloat(input.value);
        console.log(parseFloat(input.value));
        const tensorinput = [new Tensor(new Float32Array([input]), 'float32', [1,1]),];

        console.log(typeof testinput)
        console.log(typeof tensorinput)

        sess.run(tensorinput).then((output) => {
            const outputtensor = output.values().next().value;
            pred.innerHTML = outputtensor.data;
            console.log(outputtensor.data);
        });
    });

 
});

console.log('connected')