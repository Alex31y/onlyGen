import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import io
import time
import os

# --- Import specifici di HiDream ---
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# --- CONFIGURAZIONE ---
# Assicurati che il token di Hugging Face sia impostato come variabile d'ambiente su RunPod
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# --- DIZIONARIO PER TENERE I MODELLI CARICATI ---
models = {}


# --- FUNZIONE PER CARICARE I MODELLI ALL'AVVIO ---
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

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False,
                                                          token=HUGGING_FACE_HUB_TOKEN)

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


# --- FUNZIONE DI GENERAZIONE ---
def generate_image(pipe, config, prompt, resolution_str, seed):
    resolutions = {
        "1024x1024": (1024, 1024), "768x1360": (1360, 768), "1360x768": (768, 1360)
    }
    height, width = resolutions.get(resolution_str.lower(), (1024, 1024))

    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()

    generator = torch.Generator("cuda").manual_seed(seed)

    images = pipe(
        prompt=prompt, height=height, width=width,
        guidance_scale=config["guidance_scale"],
        num_inference_steps=config["num_inference_steps"],
        num_images_per_prompt=1, generator=generator
    ).images

    return images[0], seed


# --- GESTIONE AVVIO/SPEGNIMENTO API CON FASTAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Azioni all'avvio: carica i modelli
    models["hidream_pipe"], models["hidream_config"] = load_hidream_models(model_type="full")
    yield
    # Azioni alla chiusura: pulisci la VRAM
    models.clear()
    torch.cuda.empty_cache()


# --- DEFINIZIONE API ---
app = FastAPI(lifespan=lifespan)


class GenerationRequest(BaseModel):
    prompt: str
    resolution: str = "1024x1024"
    seed: int = -1


@app.get("/")
def read_root():
    return {"status": "HiDream Simple API is running, fratm!"}


@app.post("/generate")
async def create_generation(request: GenerationRequest):
    start_time = time.time()

    pipe = models.get("hidream_pipe")
    config = models.get("hidream_config")

    if not pipe or not config:
        raise HTTPException(status_code=503, detail="Modello non ancora pronto.")

    print(f"Generazione richiesta per: '{request.prompt}'")

    try:
        # 1. Genera l'immagine
        image, final_seed = generate_image(
            pipe, config, request.prompt, request.resolution, request.seed
        )

        # 2. Salva l'immagine in un buffer di memoria
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        end_time = time.time()
        print(f"Generazione completata in {round(end_time - start_time, 2)} secondi.")

        # 3. Restituisci direttamente l'immagine nella risposta HTTP
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        print(f"Errore durante la generazione: {e}")
        raise HTTPException(status_code=500, detail=str(e))