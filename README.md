# Machine Learning JS ejercicio
 Este proyecto es un ejemplo práctico de cómo aplicar machine learning básico en Node.js para análisis de datos y predicciones interactivas.
const ml = require('ml-regression'); // Librería para realizar regresión lineal
const csv = require('csvtojson'); // Para convertir CSV a JSON
const SLR = ml.SLR; // Regresión Lineal Simple


// Ruta al archivo de datos CSV
const csvFilePath = 'advertising.csv';

// Variables para almacenar datos y el modelo
let csvData = []; // Donde se guardarán los datos leídos
let X = []; // Datos de entrada (gasto en radio)
let y = []; // Datos de salida (ventas)
let regressionModel; // Variable que almacenará el modelo entrenado

// Configuramos readline para recibir datos del usuario
const readline = require('readline');
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Cargamos los datos del CSV y los convertimos a JSON
csv()
    .fromFile(csvFilePath)
    .then((jsonArray) => {
        csvData = jsonArray;
        if (csvData.length === 0) {
            console.log("Error: El archivo CSV está vacío o no tiene datos.");
            process.exit(1); // Termina el programa si no hay datos
        }
        processData(); // Procesamos los datos
        trainModel(); // Entrenamos el modelo
    })
    .catch((error) => {
        console.error("Error al leer el archivo CSV:", error);
    });

// Función para procesar y preparar los datos
function processData() {
    csvData.forEach((row) => {
        // Validamos que los valores existan y sean numéricos
        if (row.radio && row.sales) {
            X.push(parseToFloat(row.radio)); // Gasto en radio
            y.push(parseToFloat(row.sales)); // Ventas
        } else {
            console.log("Advertencia: Una fila del CSV tiene datos incompletos o inválidos y fue omitida.");
        }
    });

    // Validamos que haya suficientes datos después de limpiar el CSV
    if (X.length < 2 || y.length < 2) {
        console.error("Error: No hay suficientes datos válidos para entrenar el modelo.");
        process.exit(1);
    }
}

// Función para entrenar el modelo de regresión lineal
function trainModel() {
    regressionModel = new SLR(X, y); // Entrenamos el modelo con X y y
    console.log("Modelo de regresión lineal entrenado:");
    console.log(regressionModel.toString(3)); // Mostramos la ecuación del modelo
    startPrediction(); // Comenzamos con las predicciones
}

// Función para convertir valores a flotantes
function parseToFloat(value) {
    const parsedValue = parseFloat(value);
    if (isNaN(parsedValue)) {
        console.warn(`Advertencia: Valor no numérico encontrado: "${value}"`);
        return 0;
    }
    return parsedValue;
}

// Función para hacer predicciones interactivas
function startPrediction() {
    rl.question('Ingresa el gasto en radio para predecir ventas (CTRL+C para salir): ', (answer) => {
        const parsedInput = parseFloat(answer);
        
        // Validamos que la entrada sea un número
        if (isNaN(parsedInput)) {
            console.log("Por favor ingresa un número válido.");
        } else {
            const predictedSales = regressionModel.predict(parsedInput).toFixed(2);
            console.log(`Si se gasta ${parsedInput} en radio, se predicen ventas de aproximadamente ${predictedSales}.`);
        }

        startPrediction(); // Reinicia para permitir múltiples predicciones
    });
}

