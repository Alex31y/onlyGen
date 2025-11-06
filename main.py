import torch
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import boto3
from botocore.exceptions import NoCredentialsError
import psycopg2
import io
import time
import uuid

# --- Import specifici di HiDream ---
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# --- CONFIGURAZIONE ---
# Metti queste variabili come variabili d'ambiente per sicurezza!
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_KEY"
AWS_S3_BUCKET_NAME = "your-s3-bucket-name"
DB_CONNECTION_STRING = "dbname='yourdb' user='youruser' host='yourhost' password='yourpassword'"

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
        # Aggiungi 'dev' e 'fast' se ti servono
    }

    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)

    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path,
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer

    print("Modello caricato con successo!")
    return pipe, config


# --- FUNZIONE DI GENERAZIONE ---
def generate_image(pipe, config, prompt, resolution_str, seed):
    # Mapping risoluzioni (più pulito)
    resolutions = {
        "1024x1024": (1024, 1024),
        "768x1360": (1360, 768),  # h, w
        "1360x768": (768, 1360)  # h, w
    }
    height, width = resolutions.get(resolution_str, (1024, 1024))

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


# --- FUNZIONI PER STORAGE E DB ---
def upload_to_s3(image, bucket_name, object_name):
    # Salva l'immagine in un buffer di memoria invece che su disco
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # torna all'inizio del buffer

    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    try:
        s3_client.upload_fileobj(img_byte_arr, bucket_name, object_name)
        # Costruisci l'URL pubblico (o presigned se vuoi sicurezza)
        url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        print(f"Upload completato: {url}")
        return url
    except NoCredentialsError:
        print("Credenziali AWS non trovate.")
        return None


def save_to_db(prompt, seed, image_url, model_type):
    # Qui inserisci il record nel tuo database
    try:
        conn = psycopg2.connect(DB_CONNECTION_STRING)
        cur = conn.cursor()
        query = """
        INSERT INTO generations (id, prompt, seed, image_url, model_type, created_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        """
        # Genera un ID univoco per il record
        record_id = str(uuid.uuid4())
        cur.execute(query, (record_id, prompt, seed, image_url, model_type))
        conn.commit()
        cur.close()
        conn.close()
        print("Record salvato nel DB.")
    except Exception as e:
        print(f"Errore nel salvataggio su DB: {e}")


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
    model_type: str = "full"


@app.get("/")
def read_root():
    return {"status": "HiDream API is running, fratm!"}


@app.post("/generate")
async def create_generation(request: GenerationRequest):
    start_time = time.time()

    # Prendi il modello e la config già caricati
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

        # 2. Salva su S3 Bucket
        image_name = f"generated/{uuid.uuid4()}.png"
        image_url = upload_to_s3(image, AWS_S3_BUCKET_NAME, image_name)
        if not image_url:
            raise HTTPException(status_code=500, detail="Errore nell'upload su S3.")

        # 3. Salva i metadati sul DB
        save_to_db(request.prompt, final_seed, image_url, request.model_type)

        end_time = time.time()

        return {
            "prompt": request.prompt,
            "image_url": image_url,
            "seed": final_seed,
            "generation_time_seconds": round(end_time - start_time, 2)
        }

    except Exception as e:
        print(f"Errore durante la generazione: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Per avviare il server: uvicorn main:app --host 0.0.0.0 --port 8000