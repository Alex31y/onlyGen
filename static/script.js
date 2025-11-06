async function generateImage() {
    const promptInput = document.getElementById('prompt-input');
    const generateBtn = document.getElementById('generate-btn');
    const spinner = document.getElementById('spinner');
    const resultImage = document.getElementById('result-image');

    const prompt = promptInput.value;
    if (!prompt) {
        alert('Fratm, devi scrivere un prompt!');
        return;
    }

    // Disabilita il pulsante e mostra lo spinner
    generateBtn.disabled = true;
    generateBtn.innerText = 'Generazione in corso...';
    spinner.style.display = 'block';
    resultImage.style.display = 'none';

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                resolution: '1024x1024' // Per ora fisso, ma puoi aggiungere un dropdown
            }),
        });

        if (!response.ok) {
            // Se c'è un errore, prova a leggere il messaggio dal server
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Qualcosa è andato storto');
        }

        // Converte la risposta (che è un'immagine) in un formato visualizzabile
        const imageBlob = await response.blob();
        const imageUrl = URL.createObjectURL(imageBlob);

        // Mostra l'immagine
        resultImage.src = imageUrl;
        resultImage.style.display = 'block';

    } catch (error) {
        console.error('Errore:', error);
        alert(`Errore nella generazione: ${error.message}`);
    } finally {
        // Riabilita il pulsante e nascondi lo spinner
        generateBtn.disabled = false;
        generateBtn.innerText = 'Genera Immagine';
        spinner.style.display = 'none';
    }
}