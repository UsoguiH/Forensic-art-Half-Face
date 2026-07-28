import gc
import io
import os
import secrets
from contextlib import asynccontextmanager
from fnmatch import fnmatch
from pathlib import Path
from threading import RLock, Timer

import anyio
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import (
    Flux2KleinKVPipeline,
    Flux2Pipeline,
    Flux2Transformer2DModel,
    GGUFQuantizationConfig,
)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
    Qwen3ForCausalLM,
)

MODELS = {
    "dev": (
        "city96/FLUX.2-dev-gguf",
        "flux2-dev-Q4_K_M.gguf",
        "black-forest-labs/FLUX.2-dev",
        Flux2Pipeline,
    ),
    "klein": (
        "QuantStack/FLUX.2-Klein-9B-KV-GGUF",
        "Flux-2-Klein-9B-KV-Q4_K_M.gguf",
        "black-forest-labs/FLUX.2-klein-9b-kv",
        Flux2KleinKVPipeline,
    ),
}
IDLE_SECONDS = int(os.getenv("MODEL_IDLE_SECONDS", "900"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models")).expanduser()
if not MODEL_DIR.is_absolute():
    MODEL_DIR = Path(__file__).parent / MODEL_DIR
MODEL_DIR = MODEL_DIR.resolve()
DEV_ENCODER_REPO = "diffusers/FLUX.2-dev-bnb-4bit"

KLEIN_ENCODER = os.getenv("KLEIN_ENCODER", "4bit")
if KLEIN_ENCODER not in {"4bit", "bf16"}:
    raise ValueError("KLEIN_ENCODER must be '4bit' or 'bf16'")
PIPE = None
ACTIVE_MODEL = None
IDLE_TIMER = None
LOCK = RLock()
BASE_PATTERNS = (
    "*.json",
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "vae/*",
    "transformer/config.json",
)
MAX_REFS = 4

def env_bool(name, default):
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")

AUTO_DOWNLOAD_MODELS = env_bool("AUTO_DOWNLOAD_MODELS", True)

def validate_ref_count(count):
    if count > MAX_REFS:
        raise HTTPException(400, f"At most {MAX_REFS} reference images are allowed.")

def pipeline_images(model, refs):
    return [refs] if model == "dev" else refs

def announce(message):
    print(f"[models] {message}", flush=True)

def remote_files(repo, local_ready, label):
    """Repo file listing, or None when we're offline but the files are already local.
    Lets the server boot without network once everything has been downloaded once."""
    try:
        return HfApi().list_repo_files(repo)
    except Exception as error:
        if local_ready:
            announce(f"{label}: offline ({error}); using local files")
            return None
        raise

def model_paths(model):
    repo, filename, base, pipeline_class = MODELS[model]
    root = MODEL_DIR / model
    return repo, filename, base, pipeline_class, root / filename, root / "base"

def ensure_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not AUTO_DOWNLOAD_MODELS:
        missing = []
        for model in MODELS:
            _, _, _, _, model_path, base_path = model_paths(model)
            expected = (model_path, base_path / "model_index.json")
            missing.extend(path for path in expected if not path.is_file())
        encoder_config = MODEL_DIR / "dev" / "text_encoder_4bit" / "text_encoder" / "config.json"
        if not encoder_config.is_file():
            missing.append(encoder_config)
        if missing:
            paths = "\n  - ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"AUTO_DOWNLOAD_MODELS=false and required model files are missing:\n  - {paths}"
            )
        announce("automatic downloads disabled; using local model files")
        return

    for model in MODELS:
        repo, filename, base, _, model_path, base_path = model_paths(model)
        if model_path.is_file():
            announce(f"{model}: found {model_path}")
        else:
            announce(f"{model}: downloading GGUF {repo}/{filename}")
            hf_hub_download(repo, filename, local_dir=model_path.parent)

        patterns = BASE_PATTERNS if model == "klein" else tuple(
            pattern for pattern in BASE_PATTERNS if pattern != "text_encoder/*"
        )
        names = remote_files(base, (base_path / "model_index.json").is_file(), model)
        if names is not None:
            required = [
                name for name in names
                if any(fnmatch(name, pattern) for pattern in patterns)
            ]
            missing = [name for name in required if not (base_path / name).is_file()]
            if not missing:
                announce(f"{model}: all {len(required)} support files found in {base_path}")
            else:
                announce(f"{model}: downloading {len(missing)} missing support files from {base}")
                for name in missing:
                    announce(f"  - {name}")
                snapshot_download(base, local_dir=base_path, allow_patterns=missing)

        if model == "dev":
            encoder_path = MODEL_DIR / "dev" / "text_encoder_4bit"
            encoder_ready = (encoder_path / "text_encoder" / "config.json").is_file()
            names = remote_files(DEV_ENCODER_REPO, encoder_ready, "dev encoder")
            if names is None:
                announce(f"dev: quantized text encoder found in {encoder_path}")
            else:
                encoder_files = [n for n in names if fnmatch(n, "text_encoder/*")]
                missing_encoder = [
                    name for name in encoder_files if not (encoder_path / name).is_file()
                ]
                if missing_encoder:
                    announce(
                        f"dev: downloading {len(missing_encoder)} quantized text-encoder files "
                        f"from {DEV_ENCODER_REPO}"
                    )
                    for name in missing_encoder:
                        announce(f"  - {name}")
                    snapshot_download(
                        DEV_ENCODER_REPO,
                        local_dir=encoder_path,
                        allow_patterns=missing_encoder,
                    )
                else:
                    announce(f"dev: quantized text encoder found in {encoder_path}")
    announce("all model files are ready")

def unload_pipeline():
    global PIPE, ACTIVE_MODEL, IDLE_TIMER
    with LOCK:
        if IDLE_TIMER:
            IDLE_TIMER.cancel()
        IDLE_TIMER = None
        PIPE = None
        ACTIVE_MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_pipeline(model):
    global PIPE, ACTIVE_MODEL
    with LOCK:
        if model not in MODELS:
            raise ValueError(f"model must be one of: {', '.join(MODELS)}")
        if PIPE is not None and ACTIVE_MODEL == model:
            return PIPE
        unload_pipeline()

        _, _, _, pipeline_class, model_path, base_path = model_paths(model)
        transformer = Flux2Transformer2DModel.from_single_file(
            str(model_path),
            config=str(base_path),
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        components = {"transformer": transformer, "torch_dtype": torch.bfloat16}
        klein_4bit = model == "klein" and KLEIN_ENCODER == "4bit"
        if model == "dev":
            components["text_encoder"] = Mistral3ForConditionalGeneration.from_pretrained(
                str(MODEL_DIR / "dev" / "text_encoder_4bit"),
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
        elif klein_4bit:

            components["text_encoder"] = Qwen3ForCausalLM.from_pretrained(
                str(base_path),
                subfolder="text_encoder",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                ),
                torch_dtype=torch.bfloat16,
            )

        PIPE = pipeline_class.from_pretrained(str(base_path), **components)
        if torch.cuda.is_available():
            if model == "klein" and not klein_4bit:

                PIPE.enable_sequential_cpu_offload()
            else:
                PIPE.enable_model_cpu_offload()
            if hasattr(PIPE, "enable_vae_slicing"):
                PIPE.enable_vae_slicing()
            if hasattr(PIPE, "enable_vae_tiling"):
                PIPE.enable_vae_tiling()
        elif torch.backends.mps.is_available():
            PIPE.to("mps")
        else:
            PIPE.to("cpu")
        ACTIVE_MODEL = model
        return PIPE

def reset_idle_timer():
    global IDLE_TIMER
    if IDLE_TIMER:
        IDLE_TIMER.cancel()
    IDLE_TIMER = Timer(IDLE_SECONDS, unload_pipeline)
    IDLE_TIMER.daemon = True
    IDLE_TIMER.start()

@asynccontextmanager
async def lifespan(_app):
    ensure_models()
    yield
    unload_pipeline()

app = FastAPI(title="FLUX.2 Multi-Reference Image Server", lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "ready": PIPE is not None,
        "model": ACTIVE_MODEL,
        "multi_ref": True,
        "max_refs": MAX_REFS,
        "auto_download_models": AUTO_DOWNLOAD_MODELS,
        "idle_seconds": IDLE_SECONDS,
        "model_directory": str(MODEL_DIR),
    }

@app.post("/generate")
async def generate(
    model: str = Form(...),
    prompt: str = Form(...),
    image: list[UploadFile] | None = File(None),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(28),
    guidance: float = Form(4.0),
    seed: int = Form(-1),
):
    uploads = image or []
    validate_ref_count(len(uploads))
    if not prompt.strip():
        raise HTTPException(400, "prompt is required")
    if not 256 <= width <= 2048 or not 256 <= height <= 2048:
        raise HTTPException(400, "width and height must be between 256 and 2048")
    if width % 16 or height % 16:
        raise HTTPException(400, "width and height must be multiples of 16")
    if not 1 <= steps <= 50:
        raise HTTPException(400, "steps must be between 1 and 50")

    refs = []
    for upload in uploads:
        try:
            with Image.open(io.BytesIO(await upload.read())) as source:
                source.load()
                refs.append(source.convert("RGB"))
        except Exception as error:
            raise HTTPException(400, f"{upload.filename or 'image'} is not a valid image") from error

    seed = seed if seed >= 0 else secrets.randbelow(2**31)
    args = {
        "prompt": prompt.strip(),
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "generator": torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(seed),
    }
    if model == "dev":
        args["guidance_scale"] = guidance
    if refs:

        args["image"] = pipeline_images(model, refs)

    def run():
        with LOCK:
            image = load_pipeline(model)(**args).images[0]
            reset_idle_timer()
        return image

    try:
        result = await anyio.to_thread.run_sync(run)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(503, f"Could not load or run {model}: {error}") from error
    output = io.BytesIO()
    result.save(output, "PNG")
    return Response(output.getvalue(), media_type="image/png", headers={"X-Seed": str(seed)})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
