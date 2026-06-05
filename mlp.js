export class MLP {
    constructor(inputSize, hiddenSize,lr) { //inputSize es: [x1,x2] quindi 2, hiddenSize: n° di neuroni nascosti 
        this.lr = lr

        //crea neuroni con inputSize pesi
        this.w = []  //I neuroni sono composti da w; [[w1,w2],[]...]
        for (let i = 0; i < hiddenSize; i++) {
            this.w[i] = []
            for (let j = 0; j < inputSize; j++) {
                this.w[i][j] = Math.random() * 2 - 1
            }
        }
        this.b1 = []  // bias per neurone
        for (let i = 0; i < hiddenSize; i++) {
            this.b1[i] = Math.random() * 2 - 1
        }


        this.v = []
        for (let i = 0; i < hiddenSize; i++) {
            this.v[i] = Math.random() * 2 - 1
        }
        this.b2 = [Math.random() * 2 - 1]
    }

    predict(x){
        this.outputHidden = []  //la risposta dei neuroni del hidden

        for (let i = 0; i < this.w.length; i++) { //Somma pesata per ciascun neurone
            let sommaPesata = this.b1[i];
            let neuron = this.w[i]
            for (let j = 0; j < neuron.length; j++) {
                const w = neuron[j] 
                sommaPesata += w * x[j]
                
            }
            this.outputHidden[i] = sigmoid(sommaPesata);
        }

        // 
        this.outputFinale = this.b2[0]
        
        for (let i = 0; i < this.w.length; i++) {
            this.outputFinale += this.v[i]*this.outputHidden[i]
        }


        this.outputFinale = sigmoid(this.outputFinale)
        return this.outputFinale
    }

    train(data,target) {
        const outputPredict = this.predict(data)
        const outputError =(target - outputPredict) * dsigmoid(outputPredict);

        for (let i = 0; i < this.v.length; i++) { //Aggiorna Strato Output
            this.v[i] += this.lr * outputError * this.outputHidden[i]
        }
        this.b2[0] += this.lr*outputError


        const hiddenErrors = []
        for (let i = 0; i < this.w.length; i++) { //crea vettore errori hidden
            hiddenErrors[i] = 0
        }

        for (let i = 0; i < this.outputHidden.length; i++) { //metti gli errori
            hiddenErrors[i] = outputError * this.v[i]* dsigmoid(this.outputHidden[i]);
        }
        for (let i = 0; i < this.b1.length; i++) {
            this.b1[i] += this.lr*hiddenErrors[i]
        }

        for (let i = 0; i < this.w.length; i++) {
            let neuron = this.w[i]
            for (let j = 0; j < neuron.length; j++) {
                neuron[j] += this.lr*hiddenErrors[i]*data[j]
            }
        }



    }

}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function dsigmoid(x) {
    return x * (1 - x);
}
/*
for (let epoch = 0; epoch < 4000; epoch++) {
    for (let i = 0; i < data.length; i++) {
        mlp.train(data[i][0], data[i][1]);
    }


export class MLP {
    constructor(inputSize, hiddenSize, outputSize, lr = 0.1) {
        this.lr = lr;

        // pesi input → hidden
        this.w1 = Array.from({ length: inputSize }, () =>
            Array.from({ length: hiddenSize }, () => Math.random() * 2 - 1)
        );

        this.b1 = Array(hiddenSize).fill(0);

        // pesi hidden → output
        this.w2 = Array.from({ length: hiddenSize }, () =>
            Array.from({ length: outputSize }, () => Math.random() * 2 - 1)
        );

        this.b2 = Array(outputSize).fill(0);
    }

    // FORWARD PASS
    predict(x) {
        // hidden layer
        this.hidden = [];

        for (let j = 0; j < this.w1[0].length; j++) {
            let sum = this.b1[j];

            for (let i = 0; i < x.length; i++) {
                sum += x[i] * this.w1[i][j];
            }

            this.hidden[j] = sigmoid(sum);
        }

        // output layer
        this.output = [];

        for (let k = 0; k < this.w2[0].length; k++) {
            let sum = this.b2[k];

            for (let j = 0; j < this.hidden.length; j++) {
                sum += this.hidden[j] * this.w2[j][k];
            }

            this.output[k] = sigmoid(sum);
        }

        return this.output;
    }

    // TRAIN SINGLE SAMPLE
    train(x, y) {
        const output = this.predict(x);

        // === OUTPUT ERROR ===
        const outputErrors = output.map((o, i) => y[i] - o);

        // === OUTPUT LAYER UPDATE ===
        for (let j = 0; j < this.w2.length; j++) {
            for (let k = 0; k < this.w2[0].length; k++) {
                this.w2[j][k] += this.lr * outputErrors[k] * dsigmoid(output[k]) * this.hidden[j];
            }
        }

        for (let k = 0; k < this.b2.length; k++) {
            this.b2[k] += this.lr * outputErrors[k];
        }

        // === HIDDEN LAYER ERROR ===
        const hiddenErrors = Array(this.hidden.length).fill(0);

        for (let j = 0; j < this.hidden.length; j++) {
            for (let k = 0; k < this.w2[0].length; k++) {
                hiddenErrors[j] += outputErrors[k] * this.w2[j][k];
            }
        }

        // === INPUT → HIDDEN UPDATE ===
        for (let i = 0; i < this.w1.length; i++) {
            for (let j = 0; j < this.w1[0].length; j++) {
                this.w1[i][j] += this.lr *
                    hiddenErrors[j] *
                    dsigmoid(this.hidden[j]) *
                    x[i];
            }
        }

        for (let j = 0; j < this.b1.length; j++) {
            this.b1[j] += this.lr * hiddenErrors[j];
        }
    }
}

*/