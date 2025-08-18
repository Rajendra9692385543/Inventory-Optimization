let model;

async function loadModel() {
    model = await cocoSsd.load();
    console.log("Model loaded.");
}

loadModel();

document.getElementById("imageInput").addEventListener("change", async function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        const canvas = document.getElementById("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        const predictions = await model.detect(img);
        console.log(predictions);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "red";
        ctx.font = "16px Arial";
        ctx.fillStyle = "red";

        predictions.forEach(p => {
            ctx.beginPath();
            ctx.rect(...p.bbox);
            ctx.stroke();
            ctx.fillText(`${p.class} (${(p.score*100).toFixed(1)}%)`, p.bbox[0], p.bbox[1] > 20 ? p.bbox[1]-5 : 10);
        });
    };
});

async function addProduct() {
    const modelNo = document.getElementById("modelNo").value;
    const units = parseInt(document.getElementById("units").value);
    if (!modelNo || !units) {
        alert("Enter model number and units!");
        return;
    }
    const res = await fetch('/add_product', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model_no: modelNo, units: units})
    });
    const data = await res.json();
    alert(`Product added: ${data.model_no} (${data.units} units)`);
}
