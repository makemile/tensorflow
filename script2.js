let net;
let imgEl = document.getElementById("img");
let descEl = document.getElementById("descripcion_imagen");
const webCamElement = document.getElementById("webcam");
const classifier = knnClassifier.create();

const captureImg = async () => {
  try {
    const cam = await tf.data.webcam(webCamElement);
    return await cam.capture();
  } catch (error) {
    console.log(error);
  }
};

const app = async () => {
  try {
    console.log("Loading Model...");
    net = await mobilenet.load();
    console.log(net, "Modelo cargado correctamente");

    if (imgEl.complete) {
      const result = await net.classify(imgEl);
      displayImagePrediction(result);
    }

    imgEl.onload = async () => {
      const result = await net.classify(imgEl);
      displayImagePrediction(result);
    };

    while (true) {
      const img = await captureImg();
      const result = await net.classify(img);
      const activation = net.infer(img, "conv_preds");
      let result2;

      try {
        result2 = await classifier.predictClass(activation);
        const classes = ["Gatos", "Dino", "Elena", "Rock"];
        document.getElementById("console2:" + classes[result2.label]);
      } catch (error) {
        console.error(error);
      }
      document.getElementById("console").innerHTML =
        "prediction:" +
        result[0].className +
        " probability:" +
        result[0].probability;

      img.dispose();
      await tf.nextFrame();
    }
  } catch (err) {
    console.error("Error al cargar el modelo:", err);
  }
};

const addExample = async (id) => {
  const img = await captureImg();
  const activation = net.infer(img, true);
  classifier.addExample(activation, id);
  img.dispose();
};

const displayImagePrediction = (result) => {
  if (result) {
    descEl.innerHTML = JSON.stringify(result, null, 2);
    console.log(result);
  } else {
    console.log(
      "El modelo no se cargó o el método classify no está disponible"
    );
  }
};

let count = 0;

const cambiarImagen = async () => {
  count = count + 1;
  imgEl.src = "https://picsum.photos/200/300?random=" + count;
};

app();
