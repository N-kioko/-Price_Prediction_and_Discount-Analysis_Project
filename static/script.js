async function predictPrice() {
    const formData = new FormData(document.getElementById('predictForm'));
    const formObject = Object.fromEntries(formData);

    // Prepare the data to send in the request, with default fallback for invalid values
    const data = {
        brand: formObject.brand,
        color: formObject.color,
        screen_size: parseFloat(formObject.screen_size) || 0,  // Default to 0 if invalid
        ram: parseInt(formObject.ram) || 0,
        rom: parseInt(formObject.rom) || 0,
        warranty: parseInt(formObject.warranty) || 0,
        camera: parseInt(formObject.camera) || 0,
        battery_power: parseInt(formObject.battery_power) || 0,
        sim_count: parseInt(formObject.sim_count) || 0
    };

    // Log the data object for debugging
    console.log("Sending data:", data);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)  // Send the data as JSON
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction request failed');
        }

        const result = await response.json();
        document.getElementById('result').innerHTML = `
            <p><strong>Predicted Phone Price:</strong> ${result.price}</p>
        `;
    } catch (error) {
        console.error(error);
        document.getElementById('result').innerHTML = `
            <p>Error: ${error.message || 'Unable to fetch prediction.'}</p>
        `;
    }
}

// Add event listener for the button
document.getElementById('predictButton').addEventListener('click', predictPrice);
