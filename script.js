let model;
let stopTraining;

async function getData() {
  const dataHouseR = await fetch("data.json");
  const dataHouse = await dataHouseR.json();
  let cleanedData = dataHouse.map((house) => ({
    precio: house.Precio,
    cuartos: house.NumeroDeCuartosPromedio,
  }));

  cleanedData = cleanedData.filter(
    (house) => house.precio != null && house.cuartos != null
  );
  return cleanedData;
}

async function lookAtInferenceCurve() {
  let data = await getData();
  let tensorData = await convertDataTensor(data);
  const { inputsMax, inputsMin, labelMin, labelMax } = tensorData;

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const desnormX = xs.mul(labelMax.sub(labelMin)).add(labelMin);
    const desnormY = preds.mul(labelMax.sub(labelMin)).add(labelMin);
    return [desnormX.dataSync(), desnormY.dataSync(), desnormY.dataSync()];
  });

  const predictionPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = data.map((d) => ({
    x: d.cuartos,
    y: d.precio,
  }));

  tfvis.render.scatterplot(
    { name: "predictions vs originals" },
    { values: [originalPoints, predictionPoints], series: [] },
    {
      xLabel: "Cuartos",
      yLabel: "Precio",
      height: 300,
    }
  );
}

async function uploadModel() {
  const uploadJSONInput = document.getElementById("upload-json");
  const uploadWeightsInput = document.getElementById("upload-weights");
  model = await tf.loadLayersModel(
    tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]])
  );
  console.log("modelo cargado");
}

function visualizeData(data) {
  const values = data.map((d) => ({
    x: d.cuartos,
    y: d.precio,
  }));
  tfvis.render.scatterplot(
    { name: "Cuartos vs Precio" },
    { values: values },
    { xLabel: "Cuartos", yLabel: "Precio", height: 300 }
  );
}

function createModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [1],
      units: 1,
      useBias: true,
    })
  );
  model.add(
    tf.layers.dense({
      units: 1,
      useBias: true,
    })
  );

  return model;
}

const optimizer = tf.train.adam();
const loss_function = tf.losses.meanSquaredError;
const metric = ["mse"];

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: optimizer,
    loss: loss_function,
    metrics: metric,
  });

  const surface = { name: "show.history live", tab: "Training" };
  const sizeBatch = 28;
  const epochs = 50;
  const history = [];
  return await model.fit(inputs, labels, {
    sizeBatch,
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        history.push(log);
        tfvis.show.history(surface, history, ["loss", "mse"]);

        if (stopTraining) {
          model.stopTraining = true;
        }
      },
    },
  });
}

async function saveModel() {
  const saveResult = await model.save("downloads://model-regresion");
  return saveResult;
}

function convertDataTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);
    const inputs = data.map((d) => d.cuartos);
    const label = data.map((d) => d.precio);
    const tensorInput = tf.tensor2d(inputs, [inputs.length, 1]);
    const tensorLabel = tf.tensor2d(label, [label.length, 1]);

    const inputsMax = tensorInput.max();
    const inputsMin = tensorInput.min();
    const labelMax = tensorLabel.max();
    const labelMin = tensorLabel.min();

    const inputsNormalized = tensorInput
      .sub(inputsMin)
      .div(inputsMax.sub(inputsMin));
    const labelNormalized = tensorLabel
      .sub(labelMin)
      .div(labelMax.sub(inputsMax));

    return {
      inputs: inputsNormalized,
      label: labelNormalized,
      inputsMax,
      inputsMin,
      labelMax,
      labelMin,
    };
  });
}

async function run() {
  const data = await getData();
  visualizeData(data);
  model = createModel();
  const tensorData = convertDataTensor(data);
  const { inputs, label } = tensorData;
  trainModel(model, inputs, label);
}

run();
