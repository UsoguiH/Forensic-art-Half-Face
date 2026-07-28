import os
import gc
import glob
import numpy as np

_STT = None

def _get_stt(model_size: str = "base", device: str = "cuda"):
    global _STT
    if _STT is None:
        from faster_whisper import WhisperModel
        compute = "float16" if device == "cuda" else "int8"
        _STT = WhisperModel(model_size, device=device, compute_type=compute)
    return _STT

def transcribe_voice(audio_path: str, model_size: str = "base",
                     device: str = "cuda") -> str:
    """Return the spoken prompt as a clean text string."""
    stt = _get_stt(model_size, device)

    segments, _info = stt.transcribe(audio_path, beam_size=1, language="en")
    prompt = " ".join(seg.text for seg in segments).strip()
    return prompt

def record_microphone(seconds: float = 5.0, samplerate: int = 16000,
                      out_path: str = "prompt.wav") -> str:
    """Optional: grab audio from the mic. pip install sounddevice soundfile."""
    import sounddevice as sd
    import soundfile as sf
    print(f"Recording {seconds:.0f}s ...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate,
                   channels=1, dtype="float32")
    sd.wait()
    sf.write(out_path, audio, samplerate)
    return out_path

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"

GGUF_URL = ("https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/"
            "blob/main/flux-2-klein-4b-Q4_K_S.gguf")

_PIPE = None
_FACE_APP = None

def _get_pipe():
    """Load the official FLUX.2 Klein 4B pipeline once."""
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import Flux2KleinPipeline
        _PIPE = Flux2KleinPipeline.from_pretrained(
            KLEIN_REPO, torch_dtype=torch.bfloat16,
        )
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if free_gb >= 18:
            _PIPE.to("cuda")
        else:
            _PIPE.enable_model_cpu_offload()
        try:
            _PIPE.set_progress_bar_config(disable=True)
        except Exception:
            pass
    return _PIPE

def _get_face_app():
    """Load InsightFace (RetinaFace detect + ArcFace embed) once."""
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        _FACE_APP = FaceAnalysis(name="buffalo_l",
                                 providers=["CUDAExecutionProvider",
                                            "CPUExecutionProvider"])
        _FACE_APP.prepare(ctx_id=0, det_size=(640, 640))
    return _FACE_APP

def _embed_largest_face(app, rgb_image):
    """Return the L2-normalized ArcFace embedding of the biggest face, or None."""
    bgr = np.asarray(rgb_image)[:, :, ::-1]
    faces = app.get(bgr)
    if not faces:
        return None

    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return faces[-1].normed_embedding.astype(np.float32)

def generate_faces(prompt: str, n: int = 30, height: int = 1024, width: int = 1024,
                   steps: int = 4, base_seed: int = 0, save_dir: str | None = None):
    """Burst-generate n images for the same prompt with distinct random seeds."""
    import torch
    pipe = _get_pipe()
    images = []
    for i in range(n):
        seed = base_seed + i
        gen = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode():
            img = pipe(
                prompt=prompt,
                height=height, width=width,
                guidance_scale=1.0,
                num_inference_steps=steps,
                generator=gen,
            ).images[0]
        images.append(img)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            img.save(os.path.join(save_dir, f"face_{i:03d}_seed{seed}.png"))
        torch.cuda.empty_cache()
    return images

def reduce_to_query(images, use: str = "mean"):
    """
    Turn N generated faces into ONE robust 512-d query vector.

    This is the CORRECT replacement for pixel-averaging: variance reduction
    happens in embedding space, so the query stays sharp and specific.

      use="mean"   -> average the N embeddings, renormalize  (noise-robust centroid)
      use="medoid" -> the single real sample closest to that centroid
                      (safest if synthetic->real domain gap worries you)

    Returns: (query_vec [512], medoid_index, valid_count)
    """
    app = _get_face_app()
    embs = []
    for img in images:
        e = _embed_largest_face(app, img)
        if e is not None:
            embs.append(e)
    if not embs:
        raise RuntimeError("No faces detected in any generated image.")
    embs = np.stack(embs).astype(np.float32)

    mean_emb = embs.mean(axis=0)
    mean_emb /= (np.linalg.norm(mean_emb) + 1e-9)
    medoid_idx = int((embs @ mean_emb).argmax())

    query = mean_emb if use == "mean" else embs[medoid_idx]
    return query.astype(np.float32), medoid_idx, len(embs)

def _degrade(img, target_px: int):
    """
    Resolution-MATCH the query to a low-quality gallery on purpose:
    downsample to target_px then back up. Use this so the query does not
    out-resolve the evidence (your point about full-HD queries hurting search).
    Apply the SAME degrade to the gallery in build_index() for a common domain.
    """
    from PIL import Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    w, h = img.size
    small = img.resize((target_px, target_px), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

def aligned_average(images, upsample: int = 256, return_crops: bool = False):
    """
    The CORRECT version of your averaging idea (Galton composite).

    Each seed puts the face in a different place/scale/angle, so a raw pixel
    average smears everything. Here every face is first landmark-aligned to the
    canonical ArcFace template, THEN averaged. Effect you wanted:
      - prompt-grounded attributes are consistent  -> reinforce, stay sharp
      - unspecified / hallucinated attributes vary  -> wash out to a soft blur
    The composite is inherently low-res, which also domain-matches poor evidence.

    Returns: (composite_rgb [upsample,upsample,3] uint8, n_aligned)
             or, with return_crops=True,
             (composite_rgb, n_aligned, crops_rgb) where crops_rgb is a list of
             112x112x3 uint8 RGB canonical crops -- the SAME aligned stack the
             composite is averaged from, so a per-pixel variance map computed on
             it lines up exactly with the composite.
    """
    from insightface.utils import face_align
    from PIL import Image
    app = _get_face_app()

    crops = []
    for img in images:
        bgr = np.asarray(img)[:, :, ::-1]
        faces = app.get(bgr)
        if not faces:
            continue
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kps = faces[-1].kps
        crop = face_align.norm_crop(bgr, landmark=kps, image_size=112)
        crops.append(crop.astype(np.float32))
    if not crops:
        raise RuntimeError("No faces could be aligned.")

    avg_bgr = np.mean(np.stack(crops), axis=0).clip(0, 255).astype(np.uint8)
    avg_rgb = avg_bgr[:, :, ::-1]
    composite = np.asarray(
        Image.fromarray(avg_rgb).resize((upsample, upsample), Image.BILINEAR))
    if return_crops:
        crops_rgb = [c.clip(0, 255).astype(np.uint8)[:, :, ::-1] for c in crops]
        return composite, len(crops), crops_rgb
    return composite, len(crops)

def embed_composite(composite_rgb):
    """
    Embed the aligned-average composite for search. Tries the normal
    detect->align->embed path; if the blurry composite defeats the detector,
    falls back to the recognition sub-model on the already-canonical crop.
    (app.models['recognition'] / get_feat are insightface-version-sensitive.)
    """
    from PIL import Image
    app = _get_face_app()
    e = _embed_largest_face(app, composite_rgb)
    if e is not None:
        return e
    rec = app.models["recognition"]
    bgr112 = np.asarray(Image.fromarray(composite_rgb).resize((112, 112)))[:, :, ::-1]
    feat = np.asarray(rec.get_feat(bgr112)).flatten().astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-9)

def build_index(image_glob: str, index_path: str = "faces.index",
                ids_path: str = "faces_ids.npy", batch_report: int = 5000):
    """
    ONE-TIME preprocessing: embed every mugshot and store an exact FAISS index
    plus a parallel array mapping row -> source file path.
    """
    import faiss
    app = _get_face_app()

    paths = sorted(glob.glob(image_glob))
    if not paths:
        raise FileNotFoundError(f"No images matched: {image_glob}")

    index = faiss.IndexFlatIP(512)
    kept_ids = []
    from PIL import Image
    for k, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        e = _embed_largest_face(app, img)
        if e is None:
            continue
        index.add(e[None, :])
        kept_ids.append(p)
        if (k + 1) % batch_report == 0:
            print(f"  processed {k + 1}/{len(paths)}  (indexed {index.ntotal})")

    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(kept_ids))
    print(f"Done. Indexed {index.ntotal} faces from {len(paths)} files.")
    return index_path, ids_path

def search(query_vec, k: int = 30, index_path: str = "faces.index",
           ids_path: str = "faces_ids.npy"):
    """Return the top-k most similar real faces to the query embedding."""
    import faiss
    index = faiss.read_index(index_path)
    ids = np.load(ids_path, allow_pickle=True)
    q = np.ascontiguousarray(query_vec[None, :], dtype=np.float32)
    sims, idx = index.search(q, k)
    results = [(str(ids[i]), float(s)) for s, i in zip(sims[0], idx[0]) if i != -1]
    return results

if __name__ == "__main__":

    prompt = transcribe_voice("prompt.wav", model_size="base")
    print("Prompt:", prompt)

    imgs = generate_faces(prompt, n=30, steps=4, save_dir="generated")
    query, medoid_i, n_valid = reduce_to_query(imgs, use="mean")
    imgs[medoid_i].save("query_representative.png")
    print(f"Built query from {n_valid}/30 faces; medoid = image {medoid_i}")

    for rank, (path, sim) in enumerate(search(query, k=30), 1):
        print(f"{rank:2d}. {sim:.3f}  {path}")
