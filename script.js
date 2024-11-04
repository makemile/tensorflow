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

async function run() {
  const data = await getData();
  visualizeData(data);
  createModel();
}

run();
