
function showLoader() {
    document.getElementById("loader").style.display = "flex";
}

function hideLoader() {
    document.getElementById("loader").style.display = "none";
}


function showPreview(input, previewId, reloadBtnId = null) {
    const preview = document.getElementById(previewId);
    const reloadBtn = reloadBtnId ? document.getElementById(reloadBtnId) : null;

    if (input.files && input.files[0]) {
        preview.src = URL.createObjectURL(input.files[0]);
        preview.style.display = "block";
        if (reloadBtn) reloadBtn.style.display = "inline-block"; 
    } else {
        preview.style.display = "none";
        if (reloadBtn) reloadBtn.style.display = "none";
    }
}


document.getElementById("foodFile").addEventListener("change", function() {
    showPreview(this, "foodPreview", "reloadFoodBtn");
});
document.getElementById("foodCamera").addEventListener("change", function() {
    showPreview(this, "foodPreview", "reloadFoodBtn");
});


document.getElementById("ingredientFile").addEventListener("change", function() {
    showPreview(this, "ingredientPreview");
});
document.getElementById("ingredientCamera").addEventListener("change", function() {
    showPreview(this, "ingredientPreview");
});


function getActiveFoodInput() {
    const gallery = document.getElementById("foodFile");
    const camera = document.getElementById("foodCamera");
    return camera.files.length ? camera : gallery;
}
function getActiveIngredientsInput() {
    const gallery = document.getElementById("ingredientFile");
    const camera = document.getElementById("ingredientCamera");
    return camera.files.length ? camera : gallery;
}

let detectedFood = null;
let excludedFoods = []; // keep track of previous wrong predictions



async function detectFood() {
    const input = getActiveFoodInput();
    if (!input.files.length) {
        alert("Please select or capture an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", input.files[0]);
    formData.append("mode", "food");
    formData.append("exclude", JSON.stringify(excludedFoods));

    showLoader();

    try {
        const res = await fetch("http://192.168.1.8:5000/analyze", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        detectedFood = data.food;

        document.getElementById("foodName").innerHTML =
            `🍲 <strong>Detected Food:</strong> ${data.food} <br>
             <small>Confidence: ${(data.confidence * 100).toFixed(1)}%</small>`;

        document.getElementById("detectAllergenBtn").disabled = false;

    } catch (err) {
        alert("Server error while detecting food");
    } finally {
        hideLoader();
    }
}

// If user clicks "This is wrong", push to excluded list
function markWrongFood() {
    if (detectedFood) {
        excludedFoods.push(detectedFood); 
        detectedFood = null;
        document.getElementById("foodName").innerHTML = "";
        document.getElementById("detectAllergenBtn").disabled = true;
    }
}



async function detectAllergens() {
    if (!detectedFood) {
        alert("Please detect food first.");
        return;
    }

    const input = getActiveFoodInput();

    const formData = new FormData();
    formData.append("image", input.files[0]);
    formData.append("mode", "allergen");
    formData.append("food_name", detectedFood);

    showLoader();

    try {
        const res = await fetch("http://192.168.1.8:5000/analyze", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        displayResult(data);

    } catch (err) {
        alert("Server error while detecting allergens");
    } finally {
        hideLoader();
    }
}


async function detectAller() {
    const input = getActiveIngredientsInput();
    if (!input.files.length) {
        alert("Please select or capture an image first.");
        return;
    }
    
    const formData = new FormData();
    formData.append("image", input.files[0]);

    showLoader();

    try {
        const res = await fetch("http://192.168.1.8:5000/Ingredients", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        displayResult(data);

        if (data.error) {
            alert(data.error);
            return;
        }

    } catch (err) {
        alert("Server error while detecting Ingredient");
    } finally {
        hideLoader();
    }


}



function displayResult(data) {
    const resultBox = document.getElementById("result-text");
    const foodNameBox = document.getElementById("foodName");
    const allergenBtn = document.getElementById("detectAllergenBtn");


    if (data.error) {
        resultBox.innerHTML = `<span class="danger">❌ ${data.error}</span>`;
        return;
    }

    
    let html = "";

    
    if (data.food) {
        html += `🍲 <strong>Detected Food:</strong> ${data.food}<br><br>`;
        allergenBtn.disabled = false; 
    }

    if (!data.allergens || Object.keys(data.allergens).length === 0) {
        html += `<span class="safe">✅ No allergens detected</span>`;
    } else {
        html += `<strong>⚠️ Detected Allergens:</strong><br><br>`;
        for (const category in data.allergens) {
            html += `
                <div class="allergen-box">
                    <strong>${category.toUpperCase()}:</strong>
                    ${data.allergens[category].join(", ")}
                </div>
            `;
        }
    }

    resultBox.innerHTML = html;
}
