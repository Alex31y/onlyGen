import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import io
import time
import os

# Per servire il frontend HTML
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- Import specifici di HiDream ---
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# --- CONFIGURAZIONE ---
# Assicurati che il token di Hugging Face sia impostato come variabile d'ambiente su RunPod
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# --- DIZIONARIO PER TENERE I MODELLI CARICATI IN MEMORIA ---
models = {}


# --- FUNZIONE PER CARICARE I MODELLI ALL'AVVIO DELL'APPLICAZIONE ---
def load_hidream_models(model_type="full"):
    print(f"Caricamento del modello HiDream in corso ({model_type})...")

    MODEL_PREFIX = "HiDream-ai"
    LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    MODEL_CONFIGS = {
        "full": {
            "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
            "guidance_scale": 5.0,
            "num_inference_steps": 50,
            "shift": 3.0,
            "scheduler": FlowUniPCMultistepScheduler
        }
    }

    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME, use_fast=False, token=HUGGING_FACE_HUB_TOKEN)

    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
        token=HUGGING_FACE_HUB_TOKEN
    ).to("cuda")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        token=HUGGING_FACE_HUB_TOKEN
    ).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path,
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
        token=HUGGING_FACE_HUB_TOKEN
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer

    print("Modello caricato con successo!")
    return pipe, config


# --- FUNZIONE DI GENERAZIONE DELL'IMMAGINE ---
def generate_image(pipe, config, prompt, resolution_str, seed):
    resolutions = {
        "1024x1024": (1024, 1024),
        "768x1360": (1360, 768),  # h, w
        "1360x768": (768, 1360)  # h, w
    }
    height, width = resolutions.get(resolution_str.lower(), (1024, 1024))  # Default a 1024x1024 se non trova la chiave

    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()

    generator = torch.Generator("cuda").manual_seed(seed)

    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=config["guidance_scale"],
        num_inference_steps=config["num_inference_steps"],
        num_images_per_prompt=1,
        generator=generator
    ).images

    return images[0], seed


# --- GESTIONE AVVIO/SPEGNIMENTO API (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Azioni all'avvio: carica i modelli in memoria
    print("Avvio dell'applicazione... Caricamento modelli.")
    models["hidream_pipe"], models["hidream_config"] = load_hidream_models(model_type="full")
    yield
    # Azioni alla chiusura: pulisci la VRAM
    print("Chiusura dell'applicazione... Pulizia memoria.")
    models.clear()
    torch.cuda.empty_cache()


# --- DEFINIZIONE DELL'APPLICAZIONE FASTAPI ---
app = FastAPI(lifespan=lifespan)

# Monta la cartella 'static' per servire file come CSS, JS, e immagini
app.mount("/static", StaticFiles(directory="static"), name="static")


# Definisce il modello dei dati in ingresso per l'endpoint /generate
class GenerationRequest(BaseModel):
    prompt: str
    resolution: str = "1024x1024"
    seed: int = -1


# --- DEFINIZIONE DEGLI ENDPOINTS (LE "PAGINE" DELLA NOSTRA API) ---

# Endpoint per la pagina principale: restituisce il file index.html
@app.get("/", response_class=FileResponse)
async def read_index():
    return "static/index.html"


# Endpoint per la generazione dell'immagine
@app.post("/generate")
async def create_generation(request: GenerationRequest):
    start_time = time.time()

    pipe = models.get("hidream_pipe")
    config = models.get("hidream_config")

    if not pipe or not config:
        raise HTTPException(status_code=503, detail="Modello non ancora pronto. Riprova tra qualche istante.")

    print(f"Generazione richiesta per il prompt: '{request.prompt}'")

    try:
        # 1. Chiama la funzione per generare l'immagine
        image, final_seed = generate_image(
            pipe, config, request.prompt, request.resolution, request.seed
        )

        # 2. Salva l'immagine in un buffer di memoria (RAM) invece che su disco
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        end_time = time.time()
        print(f"Generazione completata in {round(end_time - start_time, 2)} secondi. Seed usato: {final_seed}")

        # 3. Restituisci direttamente l'immagine nella risposta HTTP
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        print(f"Errore grave durante la generazione: {e}")
        # In caso di errore, restituisci un JSON con il messaggio di errore
        raise HTTPException(status_code=500, detail=f"Errore interno del server: {str(e)}")


# Se questo file viene eseguito direttamente (es. per test locali), avvia il server uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)