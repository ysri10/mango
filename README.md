# 🥭 mango: Autonomous Multi-Agent Computer Vision Swarm

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-Apple%20Silicon%20MPS-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-blueviolet.svg)](https://ultralytics.com)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini%203.1%20Flash%20Lite-brightgreen.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An autonomous multi-agent swarm architecture that combines cloud-based vision planning (**Google Gemini 3.1 Flash Lite**) with specialized local **Ollama** agents to collaboratively architect, implement, audit, and compile production-grade, executable Jupyter Notebooks (`cv_pipeline.ipynb`) and architectural logs (`work.md`).

---

## 🌟 Key Highlights

- **🧠 Hybrid Cloud-Edge Orchestration**: Mango (Gemini 3.1 Flash Lite) oversees technical roadmaps, synthesizes distributed agent work, and writes ready-to-run Jupyter Notebooks.
- **⚡ Apple Silicon MPS Optimized**: Native hardware acceleration via PyTorch Metal Performance Shaders (`mps`) with automatic fallback to CUDA/CPU.
- **🖥️ Non-Blocking Jupyter Visualizations**: Strict avoidance of blocking `cv2.imshow()` and `cv2.waitKey()` calls that freeze notebook kernels on macOS, enforcing `%matplotlib inline` rendering.
- **🧹 Zero-Leak Mac Unified Memory**: Local Ollama vision models are dispatched with `keep_alive: 0`, flushing weights immediately after task execution to prevent GPU RAM overflow.
- **📁 Automated Pipeline Structuring**: Automatically creates dated and slugified project folders (e.g. `2026-11-05_Real_time_face_detection/`) for every unique vision task.

---

## 🏗️ Architecture & Specialized Agent Roster

```text
                                [ User Task Prompt ]
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │     Mango (Gemini 3.1 Flash Lite)     │
                     │  Master Roadmap & Pipeline Architect  │
                     └───────────────────┬───────────────────┘
                                         │
         ┌───────────────────────────────┴───────────────────────────────┐
         ▼                                                               ▼
┌──────────────────┐                                            ┌──────────────────┐
│ Optic (Qwen 3.5) │ ── Spatial & CV System Architect           │ Lens (Gemma 4)   │ ── Dataset & Run Guide
└────────┬─────────┘                                            └────────┬─────────┘
         ▼                                                               ▼
┌──────────────────┐                                            ┌──────────────────┐
│ Pixel (Qwen 1.5) │ ── CV Boilerplate & MPS Device Config      │ Cortex (Nomic)   │ ── Vector State Indexer
└────────┬─────────┘                                            └────────┬─────────┘
         ▼                                                               ▼
┌──────────────────┐                                            ┌──────────────────┐
│ Retina (Qwen 14B)│ ── Deep Vision Inference (YOLO / PyTorch)  │ Voxel (DeepSeek) │ ── Tensor & Non-blocking Audit
└────────┬─────────┘                                            └────────┬─────────┘
         ▼                                                               ▲
┌──────────────────┐                                                     │
│ Shutter (Phi-4)  │ ── Pipeline State Compressor                        │
└────────┬─────────┘                                                     │
         ▼                                                               │
┌──────────────────┐                                                     │
│ Frame (Qwen 7B)  │ ── OpenCV Preprocessing & Inline Visualizations ────┘
└──────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │     Mango (Gemini 3.1 Flash Lite)     │
                     │ Synthesizes & Compiles Final Outputs  │
                     └───────────────────┬───────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 📄 `work.md`                   📓 `cv_pipeline.ipynb`
```

| Agent | Model | Specialty & Responsibilities |
| :--- | :--- | :--- |
| **Mango** | `gemini-3.1-flash-lite-preview` | **Master Orchestrator**: Designs technical roadmap, monitors execution, and compiles final executable `.ipynb` notebook. |
| **Optic** | `qwen3-5-9b-8k` | **Spatial Architect**: Establishes image dimensions, color spaces (BGR vs. RGB), tensor layouts `(B, C, H, W)`, and cell flow. |
| **Pixel** | `qwen2.5-coder:1.5b-base` | **Boilerplate Generator**: Imports (`torch`, `cv2`, `ultralytics`, `matplotlib`), `%matplotlib inline`, and device configurations. |
| **Retina** | `qwen2-5-coder-14b-8k` | **Lead Vision Engineer**: Production-ready PyTorch & YOLO detection/segmentation engines. |
| **Shutter** | `phi4-mini-8k` | **Context Compressor**: Condenses pipeline memory in `work.md` into 4 tight bullets to save LLM context window. |
| **Frame** | `qwen-coder-8k` | **Preprocessing & Viz**: Image loading, letterboxing, BGR-to-RGB color transformation, and matplotlib visualization patches. |
| **Voxel** | `deepseek-r1-7b-8k` | **CV Auditor**: Inspects code for channel ordering bugs, tensor shape mismatches, and prevents blocking `cv2.imshow()` GUI freezes. |
| **Lens** | `gemma4-8k` | **Technical Writer**: Documents dataset formats (YOLO/COCO), camera input requirements, and setup commands. |
| **Cortex** | `nomic-embed-text:latest` | **Vector Indexer**: Creates vector embeddings of pipeline specifications in `work.md`. |

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([Get one here](https://aistudio.google.com/))
- (Optional) [Ollama](https://ollama.com/) running locally for edge agent execution

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/ysri10/mango.git
cd mango
```

Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the example environment file and configure your API key:
```bash
cp .env.example .env
export GEMINI_API_KEY="your-gemini-api-key-here"
```

---

## 🧪 Quick Smoke Test

Verify your environment, Apple Silicon GPU acceleration (MPS), OpenCV preprocessing, and Matplotlib inline visualization in 5 seconds:

```bash
python test_pipeline.py
```

Expected output:
```text
============================================================
🥭 mango CV Pipeline: Quick Smoke Test
============================================================
✅ Apple Silicon GPU Acceleration (MPS) is ACTIVE
✅ Generated synthetic 640x640 RGB image canvas
✅ Rendered non-blocking visualization saved to: test_output.png
============================================================
🎉 ALL CHECKS PASSED: Environment and CV rendering pipeline are operational!
============================================================
```

---

## 🕹️ Running the Swarm

Launch the multi-agent orchestrator:
```bash
python mango.py start agents
```

When prompted, input your target vision goal:
```text
👉 Assign the Computer Vision task/project: Real-time person detection and bounding box overlay
```

### Outputs Generated

Mango creates a dedicated, dated project folder (e.g., `2026-11-05_Real_time_person_detection/`):
- **`work.md`**: Complete architectural audit trail containing notes from Optic, Retina, Frame, Voxel, etc.
- **`cv_pipeline.ipynb`**: Ready-to-run Jupyter Notebook with full inference code and `%matplotlib inline` visualization.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
