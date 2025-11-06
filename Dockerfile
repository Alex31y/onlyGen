# Step 1: Scegliamo un'immagine di base
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Step 2: Impostiamo una cartella di lavoro
WORKDIR /app

# Step 3: INSTALLIAMO LE LIBRERIE SPECIALI
# Prima installiamo git, che serve per clonare da GitHub
RUN apt-get update && apt-get install -y git

# Ora installiamo diffusers dalla versione in sviluppo e flash-attn
RUN pip install git+https://github.com/huggingface/diffusers.git
RUN pip install -U flash-attn --no-build-isolation

# Step 4: Ora copiamo requirements.txt e installiamo il resto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copiamo tutto il resto del nostro codice
COPY . .

# Step 6: Dichiariamo la variabile d'ambiente per il token
# Il valore vero lo inserisci nell'interfaccia di RunPod!
ENV HUGGING_FACE_HUB_TOKEN=""

# Step 7: Esponiamo la porta 8000
EXPOSE 8000

# Step 8: Comando da eseguire quando il container si avvia
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]