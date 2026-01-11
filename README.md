# AI Detective

A comprehensive AI-generated content detection tool that combines multiple signal processing and forensic analysis techniques for higher accuracy detection of synthetic images, audio, video, and text.

## Features

- **Multi-Technique Detection**: Combines 7+ analysis methods for robust detection
- **Comprehensive Analysis**: Weighted ensemble scoring with confidence levels
- **Interactive Learning Lab**: Understand the science behind AI detection
- **Support for Multiple Media Types**: Images, audio, video, and text
- **Educational Visualizations**: See exactly what each technique reveals

## Quick Start

```bash
# Clone the repository
git clone https://github.com/albertoelopez/AI-Detective.git
cd AI-Detective

# Run the setup script (creates venv, installs dependencies, starts server)
./run.sh
```

Then open **http://localhost:8000** in your browser.

## Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CPU-only PyTorch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Detection Techniques

### Frequency Domain Analysis

| Technique | Media | Description |
|-----------|-------|-------------|
| **Fourier Transform (FFT)** | Image, Audio, Video | Decomposes signals into frequency components. AI content often has missing high frequencies or periodic artifacts. |
| **Wavelet Transform** | Image, Audio | Multi-resolution analysis showing both frequency content AND location. Reveals localized artifacts. |
| **Cepstral Analysis (MFCCs)** | Audio | "Spectrum of spectrum" - captures voice characteristics. Synthetic voices have different cepstral patterns. |

### Image Forensics

| Technique | Description |
|-----------|-------------|
| **Error Level Analysis (ELA)** | Analyzes JPEG compression artifacts. Different regions compress differently based on their editing history. |
| **Noise Analysis** | Camera sensors produce characteristic noise. AI images have synthetic or missing noise patterns. |
| **Histogram Analysis** | Examines color/brightness distributions. AI often has unnatural peaks, gaps, or overly smooth histograms. |

### Pattern & Motion Analysis

| Technique | Media | Description |
|-----------|-------|-------------|
| **Gradient & Edge Analysis** | Image | AI edges are often too smooth, too sharp, or have inconsistent patterns. |
| **Autocorrelation** | Image, Audio | Detects repetitive patterns from copy-paste, looping, or GAN tiling artifacts. |
| **Optical Flow** | Video | Tracks motion between frames. Deepfakes often have unnatural motion around manipulated regions. |

## App Structure

```
http://localhost:8000/                    # Simple single-technique detector
http://localhost:8000/comprehensive       # Multi-technique ensemble analysis
http://localhost:8000/learn               # Learning hub with all techniques
http://localhost:8000/learn/[technique]   # Individual technique deep-dives
```

## Comprehensive Analysis

The `/comprehensive` endpoint runs all applicable techniques simultaneously:

- **Images**: 7 techniques (FFT, Wavelet, ELA, Noise, Histogram, Gradient, Autocorrelation)
- **Audio**: 4 techniques (FFT, Wavelet, Autocorrelation, Cepstral)
- **Video**: 2 techniques (FFT, Optical Flow)

Results include:
- Combined weighted score
- Confidence level (High/Medium/Low)
- Agreement analysis between techniques
- Individual technique breakdowns
- All detected indicators with severity levels
- Switchable visualizations

## Project Structure

```
AI-Detective/
├── app/
│   └── main.py              # FastAPI application & routes
├── detectors/
│   ├── image_detector.py    # HuggingFace API + local heuristics
│   ├── audio_detector.py    # Spectral/prosody analysis
│   ├── video_detector.py    # Frame/temporal/face analysis
│   ├── text_detector.py     # Statistical analysis (with disclaimers)
│   ├── fourier_analyzer.py  # FFT analysis for all media types
│   ├── learning_analyzers.py # All educational analysis techniques
│   └── comprehensive_analyzer.py # Multi-technique ensemble
├── templates/
│   ├── index.html           # Main detector UI
│   ├── comprehensive.html   # Multi-technique analysis UI
│   ├── learn.html           # Learning hub
│   ├── learn_technique.html # Individual technique pages
│   └── fourier.html         # Fourier Transform learning page
├── static/                  # Static assets
├── uploads/                 # Temporary upload directory
├── requirements.txt         # Python dependencies
└── run.sh                   # Setup & run script
```

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Signal Processing**: NumPy, SciPy, librosa, PyWavelets
- **Computer Vision**: OpenCV, Pillow
- **Machine Learning**: PyTorch (CPU), Transformers, HuggingFace
- **Visualization**: Matplotlib
- **Frontend**: Vanilla HTML/CSS/JavaScript

## API Endpoints

### Detection
- `POST /detect/image` - Single-technique image detection
- `POST /detect/audio` - Single-technique audio detection
- `POST /detect/video` - Single-technique video detection
- `POST /detect/text` - Text analysis (form data: `text`)

### Comprehensive Analysis
- `POST /comprehensive/image` - Multi-technique image analysis
- `POST /comprehensive/audio` - Multi-technique audio analysis
- `POST /comprehensive/video` - Multi-technique video analysis

### Learning Lab
- `POST /learn/{technique}/analyze` - Individual technique analysis
- Techniques: `wavelet`, `ela`, `noise`, `histogram`, `gradient`, `autocorrelation`, `optical-flow`, `cepstral`
- `POST /fourier/{media_type}` - Fourier analysis (image/audio/video)

## Important Disclaimers

### Text Detection
AI text detection is **highly unreliable** and should NOT be used for:
- Academic integrity decisions
- Employment decisions
- Legal matters

False positives are common, especially for non-native English speakers and formal writing.

### General
This tool is for **educational and research purposes**. No detection method is 100% accurate. Results should be interpreted as indicators, not definitive proof.

## Optional Configuration

Create a `.env` file for enhanced features:

```env
# HuggingFace API token for better rate limits on image detection
HF_TOKEN=your_token_here
```

Get a free token at: https://huggingface.co/settings/tokens

## Contributing

Contributions are welcome! Areas for improvement:
- Additional detection techniques
- GPU acceleration support
- More pre-trained models
- Improved UI/UX
- API documentation

## License

MIT License - feel free to use, modify, and distribute.

## Acknowledgments

- Signal processing techniques based on academic research in digital forensics
- HuggingFace for the AI image detection model
- The open-source community for the excellent Python libraries

---

Built with Claude Code
