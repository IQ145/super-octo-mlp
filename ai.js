import {PCA} from './pca.js'
import { MLP } from './mlp.js';

let totalValues = 3

//TRIANGOLO = 0
//QUADRATO = 1
//CERCHIO = 2

let data;
let flattenedData = [];
let vectors;
let n_components = 2;

let neuronN;

let coordinatePCA;

const lrSlider = document.getElementById("lr-range");
const lrSpan = document.getElementById("lr-val");

const tfSlider = document.getElementById("tf-range");
const tfSpan = document.getElementById("tf-val");

const trainBtn = document.getElementById("train");
const negBtn = document.getElementById("negativeBtn");
const posBtn = document.getElementById("positiveBtn");
///

const allValues = []

const mlps = []
const valueLength = []
const allOutputProbabilities = []

for (let i = 0; i < totalValues; i++) {
  mlps[i] = new MLP(2, 5, 0.1)
  allValues[i] = i
}


/////////////////////////////---------------------

let points = []
let points2 = []
let allPoints = [...points, ...points2]

// === CONFIG ===
const allPlotters = document.getElementsByClassName("plotter");
const allCtx2 = []

for (let i = 0; i < allPlotters.length; i++) {
  allCtx2[i] = allPlotters[i].getContext("2d")
}

const width = 400;
const height = 400;
const padding = 40;

// === TROVA RANGE ===
let xs = allPoints.map(p => p[0]);
let ys = allPoints.map(p => p[1]);

let minX = Math.min(...xs);
let maxX = Math.max(...xs);
let minY = Math.min(...ys);
let maxY = Math.max(...ys);

// === SCALE ===
function scaleX(x) {
  if (maxX === minX) return width / 2;

  return padding +
    (x - minX) / (maxX - minX) * (width - 2 * padding);
}

function scaleY(y) {
  if (maxY === minY) return height / 2; // linea orizzontale

  return height - (
    padding +
    (y - minY) / (maxY - minY) * (height - 2 * padding)
  );
}

// === DISEGNA ASSI ===
function drawAxes2() {
  for (let i = 0; i < allPlotters.length; i++) {
    allCtx2[i].beginPath();
    allCtx2[i].strokeStyle = "#aaa";

  // asse X
    allCtx2[i].moveTo(padding, height/2);
    allCtx2[i].lineTo(width - padding, height/2);

  // asse Y
    allCtx2[i].moveTo(width/2, padding);
    allCtx2[i].lineTo(width/2, height - padding);

    allCtx2[i].stroke();
  }
}

// === DISEGNA PUNTI ===
function drawPoints(pointsArray, color, ctx2) {
  for (let p of pointsArray) {
    const x = scaleX(p[0]);
    const y = scaleY(p[1]);

    ctx2.beginPath();
    ctx2.arc(x, y, 5, 0, Math.PI * 2);
    ctx2.fillStyle = color;
    ctx2.fill();
  }
}

drawAxes2()
/////////////////////////////---------------------


let currentState = 1; //1 = positive, 0 = negative
let currentColor = "black";

let posExamples = [];
let posCount = posExamples.length;

let negExamples = [];
let negCount = negExamples.length;

// training vals;

let learningRate = 0.1;
let trainingFrequency = 2000;


async function caricaDati() {
    try {
        const response = await fetch('data.json');

        if (!response.ok) {
            throw new Error('Errore nel caricamento');
        }

        const rtuData = await response.json();

        data = rtuData
        
        for (let i = 0; i < data.length; i++) {
          const points = data[i];
            valueLength[i] = points.length
        }
        console.log(valueLength);
        
    } catch (err) {
        console.error(err);
    }
}

caricaDati()


lrSlider.oninput = function(){
    lrSpan.textContent = this.value;
    learningRate = this.value;
}


tfSlider.oninput = function(){
    tfSpan.textContent = this.value;
    trainingFrequency = this.value;
}

trainBtn.addEventListener("click", function () {

  for (let m = 0; m < mlps.length; m++) {

    let MLP = mlps[m];
    let examples = getExamples(m);

    let posExamples = examples[0].map(p => [p, [1]]);
    let negExamples = examples[1].map(n => [n, [0]]);

    let dataSet = posExamples.concat(negExamples);

    let frequency = 200;

    for (let epoch = 0; epoch < trainingFrequency; epoch++) {

      for (let j = 0; j < dataSet.length; j++) {
        MLP.train(dataSet[j][0], dataSet[j][1]);
      }

      if (epoch % frequency == 0) {

        let loss = 0;

        for (let j = 0; j < dataSet.length; j++) {
          const pred = MLP.predict(dataSet[j][0]);
          const target = dataSet[j][1][0];

          loss += (target - pred) ** 2;
        }

        loss /= dataSet.length;

        console.log("MLP " + m + " LOSS: " + loss);
      }
    }
  }

  alert("TRAINING DONE: " + trainingFrequency);
});


document.getElementById("fire").addEventListener("click", ()=>{ // disegna i punti
    let counter = 0
    for (let i = 0; i < data.length; i++) {
      const points = data[i];
      valueLength[i] = points.length
      for (let j = 0; j < points.length; j++) {
        flattenedData[counter] = points[j]
        counter++
      }
    }
    vectors = PCA.getEigenVectors(flattenedData);
    const result = PCA.computeAdjustedData(flattenedData, vectors[0], vectors[1]);
    coordinatePCA = normalizzaPunti(result.adjustedData);
    allPoints = coordinatePCA
    xs = allPoints.map(p => p[0]);
    ys = allPoints.map(p => p[1]);
    minX = Math.min(...xs);
    maxX = Math.max(...xs);
    minY = Math.min(...ys);
    maxY = Math.max(...ys);

    for (let i = 0; i < totalValues; i++) {
      let ctx2 = allCtx2[i]
      let examples = getExamples(i)
      drawPoints(examples[1], "blue", ctx2)
      drawPoints(examples[0], "red", ctx2)
    }
})

function getExamples(positiveValue){
  let startIndex = 0;
  let endIndex;

  for (let i = 0; i < positiveValue; i++) {
    startIndex += valueLength[i]
  }
  endIndex = startIndex + valueLength[positiveValue]

  let posExamples = allPoints.slice(startIndex, endIndex)
  let p1 = allPoints.slice(endIndex)
  let p2 = allPoints.slice(0,startIndex)
  let negExamples =  p2.concat(p1)

  return [posExamples, negExamples]
}

//clear
document.getElementById("clear").addEventListener("click", ()=>{
  let cells = document.getElementsByClassName("cell")
  const size = 28
  
  for (const element of cells) {
    element.classList.remove("active")
  }

  gridData = []
  for (let i = 0; i < size; i++) {
    gridData[i] = [];
    for (let j = 0; j < size; j++) {
      gridData[i][j] = 0;
    }
  }
})

//guess
document.getElementById("guess").addEventListener("click", ()=>{
    let testData = matrixToVector(gridData)
    let mediumValue = getMediumValue(flattenedData)
    let centeredTestData = centerizeData(mediumValue, [testData])

    let testDataPCA = PCA.computeAdjustedSingleData(centeredTestData, vectors[0], vectors[1]).adjustedData
  
    console.log(testDataPCA);
    
    for (let i = 0; i < totalValues; i++) {
      drawPoints([testDataPCA], "black", allCtx2[i])
      let mlp = mlps[i]
      allOutputProbabilities[i] = mlp.predict(testDataPCA)
    }

    const max = Math.max(...allOutputProbabilities);
    alert("MLP think is: " + trovaIndice(allOutputProbabilities, max) + "at: " + max +"%")
    console.log(allOutputProbabilities);
    
    //alert(guess)
    })

function trovaIndice(arr, valore) {
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] == valore) {
            return i;
        }
    }
    return -1;
}
///////////////////////////////////////grid.js

const gridElement = document.getElementById("grid")
    let mouseDown = 0;

    document.addEventListener('mousedown', () => {
      mouseDown = true;
    });

    document.addEventListener('mouseup', () => {
      mouseDown = false;
      });
    // crea la griglia 7x
    const size = 28;
    let gridData = [];

    for (let i = 0; i < size; i++) {
      gridData[i] = [];
      for (let j = 0; j < size; j++) {
        gridData[i][j] = 0;

        const cell = document.createElement("div");
        cell.classList.add("cell");

        cell.addEventListener("mouseenter", () => {
          if (mouseDown) {
            gridData[i][j] = 1;
            cell.classList.add("active");
          }
        });
        gridElement.appendChild(cell);
      }
    }

  function matrixToVector(matrix) {
    const vector = [];

    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        vector.push(matrix[i][j]);
      }
    }
    return vector;
  }

////+++++++++++++++++++//end grid.js

function normalizzaPunti(coordinatePCA) {
    let puntiNormalizzati = []
    for (let i = 0; i < coordinatePCA[0].length; i++) {
        puntiNormalizzati[i] = []
    }
    for (let i = 0; i < coordinatePCA.length; i++) {
        let xyz = coordinatePCA[i]
        for (let j = 0; j < xyz.length; j++) {
            puntiNormalizzati[j][i] = xyz[j]
        }
    }

    return puntiNormalizzati
}

function centerizeData(mediumValue, data) { // metti il mediumValue come origine 
    let finalValues = [];
    for (let i = 0; i < data.length; i++) {  // data -> [[1, 1], [3, 3], ...]
        const element = data[i];
        finalValues[i] = []
        for (let j = 0; j < element.length; j++) {   //[1, 1]     medVal = [x1, x2]
            finalValues[i][j] = element[j] - mediumValue[j]
        }
        
    }
    
    return finalValues;
}

function getMediumValue(data) {
    let finalValue = [];
    for (let i = 0; i < data.length; i++) { //data-> [[1, 1],[2,2],...]
        const element = data[i];
        for (let j = 0; j < element.length; j++) { // element -> [1, 1]
            finalValue[j] = (finalValue[j] ?? 0) + element[j];
        }
    }
    for (let i = 0; i < finalValue.length; i++) {
        finalValue[i] = finalValue[i]/data.length
        
    }


    return finalValue;
}
