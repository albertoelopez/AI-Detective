from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
from pathlib import Path

from detectors.image_detector import detect_ai_image
from detectors.audio_detector import detect_ai_audio
from detectors.video_detector import detect_ai_video
from detectors.text_detector import detect_ai_text
from detectors.fourier_analyzer import analyze_image_fft, analyze_audio_fft, analyze_video_fft
from detectors.learning_analyzers import (
    analyze_wavelet, analyze_ela, analyze_noise, analyze_histogram,
    analyze_gradient, analyze_autocorrelation, analyze_optical_flow, analyze_cepstral
)
from detectors.comprehensive_analyzer import (
    comprehensive_image_analysis, comprehensive_audio_analysis, comprehensive_video_analysis
)

app = FastAPI(title="AI Content Detector", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await detect_ai_image(str(file_path))
    os.remove(file_path)
    return result


@app.post("/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await detect_ai_audio(str(file_path))
    os.remove(file_path)
    return result


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await detect_ai_video(str(file_path))
    os.remove(file_path)
    return result


@app.post("/detect/text")
async def detect_text(text: str = Form(...)):
    result = await detect_ai_text(text)
    return result


# Fourier Transform Learning Page
@app.get("/fourier", response_class=HTMLResponse)
async def fourier_page(request: Request):
    return templates.TemplateResponse("fourier.html", {"request": request})


@app.post("/fourier/image")
async def fourier_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await analyze_image_fft(str(file_path))
    os.remove(file_path)
    return result


@app.post("/fourier/audio")
async def fourier_audio(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await analyze_audio_fft(str(file_path))
    os.remove(file_path)
    return result


@app.post("/fourier/video")
async def fourier_video(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await analyze_video_fft(str(file_path))
    os.remove(file_path)
    return result


# Learning Hub
@app.get("/learn", response_class=HTMLResponse)
async def learn_hub(request: Request):
    return templates.TemplateResponse("learn.html", {"request": request})


# Wavelet Transform
@app.get("/learn/wavelet", response_class=HTMLResponse)
async def wavelet_page(request: Request):
    return templates.TemplateResponse("learn_technique.html", {
        "request": request,
        "technique": "wavelet",
        "title": "Wavelet Transform",
        "description": "Multi-resolution analysis showing both frequency content and spatial/temporal location",
        "media_types": ["image", "audio"],
        "icon": "W",
        "color": "#7c3aed"
    })


@app.post("/learn/wavelet/analyze")
async def wavelet_analyze(file: UploadFile = File(...), media_type: str = Form("image")):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await analyze_wavelet(str(file_path), media_type)
    os.remove(file_path)
    return result


# Error Level Analysis
@app.get("/learn/ela", response_class=HTMLResponse)
async def ela_page(request: Request):
    return templates.TemplateResponse("learn_technique.html", {
        "request": request,
        "technique": "ela",
        "title": "Error Level Analysis (ELA)",
        "description": "Detect image manipulation through JPEG compression artifact analysis",
        "media_types": ["image"],
        "icon": "E",
        "color": "#dc2626"
    })


@app.post("/learn/ela/analyze")
async def ela_analyze(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await analyze_ela(str(file_path))
    os.remove(file_path)
    return result


# Noise Analysis
@app.get("/learn/noise", response_class=HTMLResponse)
async def noise_page(request: Request):
    return templates.TemplateResponse("learn_technique.html", {
        "request": request,
        "technique": "noise",
        "title": "Noise Analysis",
        "description": "Analyze image noise patterns to detect synthetic generation",
        "media_types": ["image"],
        "icon": "N",
        "color": "#dc2626"
    })


@app.post("/learn/noise/analyze")
async def noise_analyze(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await analyze_noise(str(file_path))
    os.remove(file_path)
    return result


# Histogram Analysis
@app.get("/learn/histogram", response_class=HTMLResponse)
async def histogram_page(request: Request):
    return templates.TemplateResponse("learn_technique.html", {
        "request": request,
        "technique": "histogram",
        "title": "Histogram Analysis",
        "description": "Examine color and brightness distributions for manipulation clues",
        "media_types": ["image"],
        "icon": "H",
        "color": "#dc2626"
    })


@app.post("/learn/histogram/analyze")
async def histogram_analyze(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await analyze_histogram(str(file_path))
    os.remove(file_path)
    return result


# Gradient/Edge Analysis
@app.get("/learn/gradient", response_class=HTMLResponse)
async def gradient_page(request: Request):
    return templates.TemplateResponse("learn_technique.html", {
        "request": request,
        "technique": "gradient",
        "title": "Gradient & Edge Analysis",
        "description": "Analyze edge patterns and gradients for AI artifacts",
        "media_types": ["image"],
        "icon": "G",
        "color": "#059669"
    })


@app.post("/learn/gradient/analyze")
async def gradient_analyze(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await analyze_gradient(str(file_path))
    os.remove(file_path)
    return result


# Autocorrelation
@app.get("/learn/autocorrelation", response_class=HTMLResponse)
async def autocorrelation_page(request: Request):
    return templates.TemplateResponse("learn_technique.html", {
        "request": request,
        "technique": "autocorrelation",
        "title": "Autocorrelation",
        "description": "Detect repetitive patterns and copy-paste artifacts",
        "media_types": ["image", "audio"],
        "icon": "A",
        "color": "#059669"
    })


@app.post("/learn/autocorrelation/analyze")
async def autocorrelation_analyze(file: UploadFile = File(...), media_type: str = Form("image")):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await analyze_autocorrelation(str(file_path), media_type)
    os.remove(file_path)
    return result


# Optical Flow
@app.get("/learn/optical-flow", response_class=HTMLResponse)
async def optical_flow_page(request: Request):
    return templates.TemplateResponse("learn_technique.html", {
        "request": request,
        "technique": "optical-flow",
        "title": "Optical Flow",
        "description": "Track motion patterns in video to detect deepfakes",
        "media_types": ["video"],
        "icon": "O",
        "color": "#2563eb"
    })


@app.post("/learn/optical-flow/analyze")
async def optical_flow_analyze(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await analyze_optical_flow(str(file_path))
    os.remove(file_path)
    return result


# Cepstral Analysis
@app.get("/learn/cepstral", response_class=HTMLResponse)
async def cepstral_page(request: Request):
    return templates.TemplateResponse("learn_technique.html", {
        "request": request,
        "technique": "cepstral",
        "title": "Cepstral Analysis (MFCCs)",
        "description": "Analyze voice characteristics to detect synthetic speech",
        "media_types": ["audio"],
        "icon": "C",
        "color": "#d97706"
    })


@app.post("/learn/cepstral/analyze")
async def cepstral_analyze(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await analyze_cepstral(str(file_path))
    os.remove(file_path)
    return result


# Comprehensive Multi-Technique Analysis
@app.get("/comprehensive", response_class=HTMLResponse)
async def comprehensive_page(request: Request):
    return templates.TemplateResponse("comprehensive.html", {"request": request})


@app.post("/comprehensive/image")
async def comprehensive_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await comprehensive_image_analysis(str(file_path))
    os.remove(file_path)
    return result


@app.post("/comprehensive/audio")
async def comprehensive_audio(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await comprehensive_audio_analysis(str(file_path))
    os.remove(file_path)
    return result


@app.post("/comprehensive/video")
async def comprehensive_video(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = await comprehensive_video_analysis(str(file_path))
    os.remove(file_path)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
