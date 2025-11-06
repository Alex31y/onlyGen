# Step 1: Immagine di base con PyTorch e CUDA
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Step 2: Cartella di lavoro
WORKDIR /app

# Step 3: Installazione dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copia del codice
COPY . .

# Step 5: Dichiariamo SOLO la variabile d'ambiente che ci serve
ENV HUGGING_FACE_HUB_TOKEN=""

# Step 6: Esponiamo la porta 8000
EXPOSE 8000

# Step 7: Comando di avvio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]