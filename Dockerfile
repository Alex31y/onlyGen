# Step 1: Scegliamo un'immagine di base che ha già Python, PyTorch e CUDA installati
# Questo ci risparmia un sacchio di tempo e problemi!
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Step 2: Impostiamo una cartella di lavoro all'interno del container
WORKDIR /app

# Step 3: Copiamo SOLO il file dei requisiti e installiamo le dipendenze.
# Questo è un trucco per velocizzare i build futuri: se non cambi requirements.txt, Docker non riesegue questo step.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Ora copiamo tutto il resto del nostro codice (main.py, etc.)
COPY . .

# Step 5: Dichiariamo le variabili d'ambiente. RunPod le inserirà qui.
# NON METTERE I TUOI SEGRETI QUI! Sono solo placeholder.
ENV HUGGING_FACE_HUB_TOKEN=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_S3_BUCKET_NAME=""
ENV DB_CONNECTION_STRING=""

# Step 6: Esponiamo la porta su cui girerà la nostra API
EXPOSE 8000

# Step 7: Il comando da eseguire quando il container si avvia.
# Avvia il server uvicorn e lo rende accessibile dall'esterno del container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]