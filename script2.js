let imgEl = document.getElementById("img");
let descEl = document.getElementById("descripcion_imagen");

async function app() {
  try {
    console.log("Loading Model...");
    const net = await mobilenet.load();
    console.log(net, "Modelo cargado correctamente");

    imgEl.onload = async function () {
      const result = await net.classify(imgEl);
      displayImagePrediction(result);
    };

    if (imgEl.complete) {
      const result = await net.classify(imgEl);
      displayImagePrediction(result);
    }
  } catch (err) {
    console.error("Error al cargar el modelo:", err);
  }
}

async function displayImagePrediction(result) {
  if (result) {
    descEl.innerHTML = JSON.stringify(result, null, 2);
    console.log(result);
  } else {
    console.log(
      "El modelo no se cargó o el método classify no está disponible"
    );
  }
}

let count = 0;

async function cambiarImagen() {
  count = count + 1;
  imgEl.src = "https://picsum.photos/200/300?random=" + count;
}

app();
