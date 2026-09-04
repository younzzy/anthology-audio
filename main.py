from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
from pydub import AudioSegment
import tempfile, os, uuid, requests

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "clips-temp")


@app.get("/health")
def health():
    return {"status": "ok"}


class PeaksRequest(BaseModel):
    source_url: str
    window_ms: int = 500
    min_gap_s: float = 3.0
    max_duration_s: float = 86400  # 24h max (stream marathon)


@app.post("/detect-peaks")
def detect_peaks(req: PeaksRequest):
    # 1. Vérifier la durée AVANT de télécharger quoi que ce soit
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(req.source_url, download=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"lien_invalide_ou_prive: {e}")

    duration = info.get("duration") or 0
    if duration > req.max_duration_s:
        raise HTTPException(status_code=400, detail="video_trop_longue")

    # 2. Extraire UNIQUEMENT l'audio (léger, marche même pour un stream de plusieurs heures)
    tmp_base = tempfile.mktemp()
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_base + ".%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([req.source_url])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"extraction_audio_echouee: {e}")

    audio_path = tmp_base + ".mp3"
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail="extraction_audio_echouee")

    try:
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)

        levels = []
        for start_ms in range(0, duration_ms, req.window_ms):
            window = audio[start_ms:start_ms + req.window_ms]
            levels.append({"time_s": start_ms / 1000, "energy_db": window.dBFS})

        # Seuil RELATIF : les 8% des moments les plus forts de CETTE vidéo précisément
        # (au lieu d'un seuil fixe qui rate les streams globalement calmes)
        sorted_levels = sorted([l["energy_db"] for l in levels if l["energy_db"] > -100])
        if not sorted_levels:
            return {"peaks": [], "duration_s": duration_ms / 1000, "note": "stream_silencieux"}
        threshold_db = sorted_levels[int(len(sorted_levels) * 0.92)]

        raw_peaks = [l for l in levels if l["energy_db"] >= threshold_db]

        # Regrouper les pics rapprochés, ignorer les pics isolés de moins de 1 seconde
        # (souvent du bruit parasite plutôt qu'un vrai moment fort)
        merged, group = [], []
        for p in raw_peaks:
            if group and (p["time_s"] - group[-1]["time_s"]) > (req.window_ms / 1000) * 1.5:
                if len(group) * (req.window_ms / 1000) >= 1.0:
                    merged.append(max(group, key=lambda x: x["energy_db"]))
                group = []
            group.append(p)
        if group and len(group) * (req.window_ms / 1000) >= 1.0:
            merged.append(max(group, key=lambda x: x["energy_db"]))

        final_peaks = []
        for p in merged:
            if not final_peaks or (p["time_s"] - final_peaks[-1]["time_s"]) >= req.min_gap_s:
                final_peaks.append(p)

        if not final_peaks:
            return {"peaks": [], "duration_s": duration_ms / 1000, "note": "stream_calme_aucun_pic_net"}

        return {"peaks": final_peaks, "duration_s": duration_ms / 1000}
    finally:
        os.remove(audio_path)


class ClipRequest(BaseModel):
    source_url: str
    start_seconds: float
    end_seconds: float


@app.post("/extract-clip")
def extract_clip(req: ClipRequest):
    """Télécharge SEULEMENT le passage demandé (pas tout le stream) et l'héberge
    temporairement sur Supabase Storage pour que Shotstack puisse le récupérer."""
    filename = f"{uuid.uuid4()}.mp4"
    tmp_path = tempfile.mktemp(suffix=".mp4")

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": tmp_path,
        "download_sections": [f"*{req.start_seconds}-{req.end_seconds}"],
        "quiet": True,
        "noplaylist": True,
        "force_keyframes_at_cuts": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([req.source_url])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"extraction_passage_echouee: {e}")

    if not os.path.exists(tmp_path):
        raise HTTPException(status_code=400, detail="extraction_passage_echouee")

    with open(tmp_path, "rb") as f:
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        resp = requests.post(
            upload_url,
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "video/mp4",
            },
            data=f.read(),
        )
    os.remove(tmp_path)

    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail="echec_envoi_supabase")

    return {"clip_url": f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"}