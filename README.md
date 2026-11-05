# mango: Autonomous Multi-Agent Computer Vision Swarm

An autonomous multi-agent swarm architecture that combines cloud-based vision planning (Google Gemini 3.1 Flash Lite) with specialized local Ollama agents to collaboratively architect, implement, audit, and compile production-grade, executable Jupyter Notebooks (`cv_pipeline.ipynb`) and architectural logs (`work.md`).

---

## System Overview

- **Hybrid Cloud-Edge Orchestration**: Mango (Gemini 3.1 Flash Lite) oversees technical roadmaps, synthesizes distributed agent outputs, and compiles runnable Jupyter Notebooks.
- **Apple Silicon MPS Acceleration**: Native hardware acceleration via PyTorch Metal Performance Shaders (`mps`) with automatic fallback to CUDA and CPU.
- **Non-Blocking Inline Visualizations**: Enforces non-blocking `%matplotlib inline` rendering to eliminate GUI thread deadlocks associated with native `cv2.imshow()` calls on macOS.
- **Unified Memory Lifecycle Management**: Local Ollama vision models are invoked with `keep_alive: 0`, unloading model weights immediately upon task completion to preserve host memory.
- **Automated Pipeline Structuring**: Generates isolated, timestamped directories (e.g., `2026-11-05_Real_time_face_detection/`) for each vision goal.

---

## Architecture and Agent Roster

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
                     `work.md`                  `cv_pipeline.ipynb`
```

| Agent | Model | Role and Primary Responsibilities |
| :--- | :--- | :--- |
| **Mango** | `gemini-3.1-flash-lite-preview` | **Master Orchestrator**: Formulates technical roadmaps, synthesizes agent contributions, and compiles the final executable `.ipynb` notebook. |
| **Optic** | `qwen3-5-9b-8k` | **Spatial Architect**: Defines input dimensions, color space standards (BGR vs. RGB), tensor representations `(B, C, H, W)`, and execution order. |
| **Pixel** | `qwen2.5-coder:1.5b-base` | **Boilerplate Generator**: Emits standard imports (`torch`, `cv2`, `ultralytics`, `matplotlib`), `%matplotlib inline`, and device configurations. |
| **Retina** | `qwen2-5-coder-14b-8k` | **Lead Vision Engineer**: Implements production-ready PyTorch and Ultralytics YOLO inference routines. |
| **Shutter** | `phi4-mini-8k` | **Context Compressor**: Condenses pipeline specifications into concise bullet points within `work.md` to conserve context windows. |
| **Frame** | `qwen-coder-8k` | **Preprocessing and Visualization**: Handles image ingestion, letterboxing, BGR-to-RGB conversion, and Matplotlib patch overlays. |
| **Voxel** | `deepseek-r1-7b-8k` | **CV Auditor**: Analyzes implementations for channel order bugs, tensor dimensionality mismatches, and blocking GUI calls. |
| **Lens** | `gemma4-8k` | **Technical Writer**: Documents input requirements, dataset formats (YOLO/COCO), and operational instructions. |
| **Cortex** | `nomic-embed-text:latest` | **Vector Indexer**: Computes vector embeddings of pipeline specifications stored in `work.md`. |

---

## Installation and Setup

### 1. Prerequisites

- Python 3.10 or higher
- Google Gemini API key
- (Optional) Local [Ollama](https://ollama.com/) instance for edge model execution

### 2. Repository Setup

Clone the repository and enter the project directory:
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

Define your Gemini API key in your shell environment:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Alternatively, copy the example environment file:
```bash
cp .env.example .env
```

---

## Verification Test

Verify the local environment, Apple Silicon GPU acceleration (MPS), OpenCV image ingestion, and Matplotlib inline visualization:

```bash
python test_pipeline.py
```

Expected output:
```text
============================================================
mango CV Pipeline: Smoke Test
============================================================
[PASS] Apple Silicon GPU Acceleration (MPS) is ACTIVE
[PASS] Generated synthetic 640x640 RGB image canvas
[PASS] Rendered non-blocking visualization saved to: test_output.png
============================================================
SUCCESS: Environment and CV rendering pipeline are operational.
============================================================
```

---

## Running the Swarm

Execute the orchestrator:
```bash
python mango.py start agents
```

When prompted, specify the target computer vision objective:
```text
Assign the Computer Vision task/project: Real-time person detection and bounding box overlay
```

### Generated Artifacts

Each run generates an isolated directory named with the execution date and slugified task name (e.g., `2026-11-05_Real_time_person_detection/`):
- **`work.md`**: Complete architectural design log with specifications and auditor checkmarks from each agent.
- **`cv_pipeline.ipynb`**: Fully runnable Jupyter Notebook containing end-to-end model inference and inline visualizations.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
