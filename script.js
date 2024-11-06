let model;

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
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
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
