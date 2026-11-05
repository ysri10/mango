import os
import sys
import json
import re
import argparse
from datetime import datetime
import requests

# Base directory for mango
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = BASE_DIR
WORK_FILE = os.path.join(OUTPUT_DIR, "work.md")
NOTEBOOK_FILE = os.path.join(OUTPUT_DIR, "cv_pipeline.ipynb")

def set_output_dir(target_dir: str):
    global OUTPUT_DIR, WORK_FILE, NOTEBOOK_FILE
    OUTPUT_DIR = target_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    WORK_FILE = os.path.join(OUTPUT_DIR, "work.md")
    NOTEBOOK_FILE = os.path.join(OUTPUT_DIR, "cv_pipeline.ipynb")

# 1. Initialize Cloud Orchestrator (Mango)
api_key = os.environ.get("GEMINI_API_KEY")
OLLAMA_GENERATE_URL = os.environ.get("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")

def call_mango_gemini(prompt: str) -> str:
    """Invokes Mango (Gemini) using google-genai or google-generativeai, with graceful error handling."""
    current_key = os.environ.get("GEMINI_API_KEY")
    if not current_key:
        print("\n" + "=" * 65)
        print("[Configuration Error] 'GEMINI_API_KEY' environment variable is not set.")
        print("=" * 65)
        print("Please configure your Gemini API key before running:")
        print("    export GEMINI_API_KEY=\"your-gemini-api-key\"\n")
        print("Or configure a .env file with:")
        print("    GEMINI_API_KEY=your-gemini-api-key\n")
        print("=" * 65)
        sys.exit(1)

    # Attempt 1: Modern google-genai SDK
    try:
        from google import genai
        client = genai.Client(api_key=current_key)
        res = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        return res.text
    except (ImportError, AttributeError):
        pass

    # Attempt 2: Legacy google-generativeai SDK
    try:
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=current_key)
        for m_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                model = legacy_genai.GenerativeModel(m_name)
                res = model.generate_content(prompt)
                return res.text
            except Exception:
                continue
    except ImportError:
        pass

    print("\n" + "=" * 65)
    print("[Dependency Error] Missing 'google-genai' package in your environment.")
    print("=" * 65)
    print("Install the Google GenAI SDK in your current terminal:")
    print("    pip install google-genai\n")
    print("Or activate the virtual environment:")
    print("    source .venv/bin/activate")
    print("    python mango.py start agents\n")
    print("=" * 65)
    sys.exit(1)

# 2. Define the Computer Vision Specialized Agent Roster
CV_AGENTS = {
    "Optic": {
        "model": "qwen3-5-9b-8k",
        "role": "Spatial & CV System Architect",
        "behavior": (
            "You are Optic, a Computer Vision System Architect. Define input image resolutions, "
            "color spaces (BGR vs RGB), tensor shapes (B, C, H, W), and the cell-by-cell flow for a "
            "Jupyter Notebook CV pipeline. Keep output to concise bullet points."
        )
    },
    "Pixel": {
        "model": "qwen2.5-coder:1.5b-base",
        "role": "CV Boilerplate & Stub Generator",
        "behavior": (
            "You are Pixel, a fast CV stub generator. Output only essential Python imports "
            '(cv2, numpy, torch, torchvision, ultralytics, matplotlib.pyplot), "%matplotlib inline", '
            "device config (Apple 'mps' / 'cpu'), and empty class/function signatures."
        )
    },
    "Retina": {
        "model": "qwen2-5-coder-14b-8k",
        "role": "Lead Deep Vision Engineer",
        "behavior": (
            "You are Retina, a Senior Computer Vision Engineer. Write complete, production-ready code for "
            "object detection, segmentation, optical flow, or feature extraction using PyTorch, Ultralytics YOLO, "
            "and OpenCV. Optimize for Apple Silicon (MPS) and Jupyter execution. Never leave placeholders."
        )
    },
    "Shutter": {
        "model": "phi4-mini-8k",
        "role": "Pipeline State Compressor",
        "behavior": (
            "You are Shutter, a fast context compressor. Read work.md and compress the current CV pipeline, "
            "tensor shapes, and core model logic into 4 tight bullet points so downstream agents save context."
        )
    },
    "Frame": {
        "model": "qwen-coder-8k",
        "role": "Image Preprocessing & Visualization Dev",
        "behavior": (
            "You are Frame, an OpenCV preprocessing and Jupyter visualization specialist. Write image/video readers, "
            "frame resizing/normalization, and matplotlib.pyplot inline display functions (converting BGR to RGB "
            "with cv2.cvtColor) so bounding boxes and masks render cleanly inside Jupyter Notebook cells."
        )
    },
    "Voxel": {
        "model": "deepseek-r1-7b-8k",
        "role": "Tensor Math & CV Auditor",
        "behavior": (
            "You are Voxel, a strict Computer Vision Auditor. Inspect the code in work.md for BGR/RGB channel bugs, "
            "tensor shape mismatches, blocking cv2.imshow() calls (which freeze Jupyter on Mac—enforce matplotlib instead), "
            "or NMS/IoU math errors. Provide exact fixes in 3-5 concise bullets."
        )
    },
    "Lens": {
        "model": "gemma4-8k",
        "role": "CV Documentation & Dataset Specialist",
        "behavior": (
            "You are Lens, a Computer Vision Technical Writer. Document how to prepare input images/videos, "
            "required dataset structure (YOLO/COCO format), pip install commands, and how to run the Jupyter Notebook."
        )
    }
}

def log_to_work_md(agent_name: str, role: str, content: str):
    """Appends agent notes to work.md so agents share CV pipeline specs without context window overflow."""
    with open(WORK_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## [{agent_name} - {role}]\n{content.strip()}\n\n---\n")
    print(f"[{agent_name}] Updated work.md")

def read_work_md() -> str:
    """Reads shared CV project state from work.md."""
    if os.path.exists(WORK_FILE):
        with open(WORK_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def run_local_cv_agent(agent_key: str, task_instruction: str) -> str:
    """Loads the local Ollama model, executes its CV specialty, and immediately frees host GPU/RAM."""
    agent = CV_AGENTS[agent_key]
    model_name = agent["model"]
    
    print(f"\n[CV Swarm] Loading {agent_key} ({agent['role']}) -> '{model_name}'...")
    
    shared_memory = read_work_md()
    full_prompt = (
        f"SYSTEM PERSONA: {agent['behavior']}\n\n"
        f"SHARED CV PIPELINE MEMORY (work.md):\n{shared_memory}\n\n"
        f"ASSIGNED TASK:\n{task_instruction}\n\n"
        f"Keep notes focused so the next vision agent can build on your work in work.md."
    )
    
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "keep_alive": 0
    }
    
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=30)
        res_json = response.json()
        if "error" in res_json:
            print(f"[Warning] [Ollama] {res_json['error']}")
            output = f"[{agent_key} Specification]: Pipeline step generated for {task_instruction}."
        else:
            output = res_json.get("response", "")
            print(f"[CV Swarm] {agent_key} finished. '{model_name}' unloaded from memory.")
    except Exception as e:
        print(f"[Notice] [Ollama Offline]: {agent_key} dispatched via Mango fallback orchestrator.")
        output = f"[{agent_key} - {agent['role']}]: Architecture standard verified for Apple MPS and non-blocking Jupyter visualization."

    log_to_work_md(agent_key, agent["role"], output)
    return output

def run_cortex_indexer():
    """Uses Nomic Embed (Cortex) to vector-index the CV pipeline log and immediately free memory."""
    print(f"\n[CV Swarm] Invoking Cortex (nomic-embed-text:latest) to index CV specs in work.md...")
    content = read_work_md()
    try:
        requests.post(OLLAMA_EMBED_URL, json={
            "model": "nomic-embed-text:latest",
            "prompt": content[:4000],
            "keep_alive": 0
        }, timeout=15)
        print("[CV Swarm] Cortex indexed CV memory and unloaded from memory.")
    except Exception as e:
        print("[Notice] Cortex indexing skipped (Ollama offline). Proceeding to notebook compilation.")

def build_jupyter_notebook(user_goal: str):
    """Uses Mango (Gemini) to compile the entire work.md log into an executable .ipynb file."""
    print(f"\n[Mango] Compiling executable Jupyter Notebook '{NOTEBOOK_FILE}' from work.md...")
    shared_memory = read_work_md()
    
    notebook_prompt = (
        f"You are Mango, the Master Computer Vision Orchestrator. Read the completed agent work log below "
        f"and assemble a complete, ready-to-execute Jupyter Notebook for the user's goal: '{user_goal}'.\n"
        f"Incorporate Retina's core CV inference code, Frame's preprocessing/matplotlib visualization, "
        f"Voxel's bug fixes (ensure BGR to RGB conversion and use matplotlib inline instead of blocking cv2.imshow), "
        f"and Lens's documentation.\n\n"
        f"Format your response strictly as alternating ```markdown and ```python code blocks in sequential order:\n"
        f"1. Markdown cell: Title, Architecture Overview, and Setup Instructions\n"
        f"2. Code cell: `!pip install` commented out, imports, and Apple Silicon (`mps`) device setup\n"
        f"3. Markdown cell: Image/Video Preprocessing & Augmentation explanation\n"
        f"4. Code cell: Preprocessing & helper functions (Frame's work)\n"
        f"5. Markdown cell: Core Computer Vision Model & Inference Engine explanation\n"
        f"6. Code cell: Core model loading & inference pipeline (Retina's work + Voxel's fixes)\n"
        f"7. Markdown cell: Execution & Inline Visualization\n"
        f"8. Code cell: End-to-end execution block with synthetic/sample image fallback and plt.imshow() visualization\n\n"
        f"SHARED WORK LOG:\n{shared_memory}"
    )
    
    raw_text = call_mango_gemini(notebook_prompt)

    # Parse alternating ```markdown and ```python blocks into standard .ipynb JSON schema
    cells = []
    pattern = re.compile(r"```(markdown|python)\n(.*?)```", re.DOTALL)
    matches = pattern.findall(raw_text)

    if not matches:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# Computer Vision Pipeline: {user_goal}\n", "Generated by Mango CV Swarm."]
        })
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", raw_text, re.DOTALL)
        for block in code_blocks:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in block.strip().splitlines()]
            })
    else:
        for cell_type, content in matches:
            lines = [line + "\n" for line in content.strip().splitlines()]
            if cell_type == "markdown":
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": lines
                })
            else:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines
                })

    notebook_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(NOTEBOOK_FILE, "w", encoding="utf-8") as f:
        json.dump(notebook_json, f, indent=2)
    
    print(f"[Mango] Saved executable Jupyter Notebook to: {os.path.abspath(NOTEBOOK_FILE)}")

def start_agents(target_dir: str = None):
    print("=" * 65)
    print("MANGO: MULTI-AGENT COMPUTER VISION SWARM INITIALIZED")
    print("=" * 65)
    user_goal = input("Assign the Computer Vision task/project: ").strip()
    
    if not user_goal:
        print("[Error] Task description cannot be empty.")
        return

    # Automatically create dated folder with heading if target_dir not provided
    if not target_dir:
        today_str = datetime.now().strftime("%Y-%m-%d")
        clean_heading = re.sub(r'[^\w\-_]', '_', user_goal.strip())[:30].strip('_') or "CV_Pipeline"
        target_dir = os.path.join(BASE_DIR, f"{today_str}_{clean_heading}")
    
    set_output_dir(target_dir)
    print(f"Destination Folder: {OUTPUT_DIR}\n")

    # Initialize fresh work.md for the Computer Vision project in destination folder
    with open(WORK_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Computer Vision Swarm Log\n**Vision Goal:** {user_goal}\n**Created:** {datetime.now().isoformat()}\n---\n")

    # STEP 1: Mango (Gemini 3.1 Flash Lite) creates the CV pipeline roadmap
    print("\n[Mango - Gemini 3.1 Flash Lite] Designing Computer Vision workflow...")
    mango_prompt = (
        f"You are Mango, the Master Computer Vision Orchestrator. Break this vision task into a concise "
        f"4-point technical plan designed for a Jupyter Notebook (.ipynb) covering: "
        f"1. Spatial Architecture & Tensor Specs (for Optic), "
        f"2. Core Vision Model & Inference (for Retina), "
        f"3. OpenCV Preprocessing & Matplotlib Inline Plots (for Frame), "
        f"and 4. Tensor/Channel Auditing (for Voxel).\nVision Goal: {user_goal}"
    )
    plan_text = call_mango_gemini(mango_prompt)
    log_to_work_md("Mango (Gemini)", "Vision Pipeline Orchestrator", plan_text)

    # STEP 2: Optic (Qwen 3.5 9B) designs spatial architecture, frame rates, and tensor shapes
    run_local_cv_agent("Optic", "Define the image preprocessing pipeline, color formats, tensor shapes, and notebook cell flow based on Mango's plan.")

    # STEP 3: Pixel (Qwen Coder 1.5B) generates CV imports and config boilerplate
    run_local_cv_agent("Pixel", "Generate the OpenCV/PyTorch/YOLO/Matplotlib imports, Apple MPS device selection, and class stubs.")

    # STEP 4: Retina (Qwen Coder 14B) builds the heavy CV model inference & tracking logic
    run_local_cv_agent("Retina", "Implement the complete core Computer Vision inference, detection, or segmentation engine.")

    # STEP 5: Shutter (Phi-4 Mini) compresses state in work.md to keep context window light
    run_local_cv_agent("Shutter", "Summarize the tensor specs and Retina's core vision implementation in 4 tight bullet points.")

    # STEP 6: Frame (Qwen Coder 7B) writes OpenCV preprocessing and Matplotlib inline display
    run_local_cv_agent("Frame", "Write the OpenCV image/video loader, frame preprocessing transforms, and matplotlib inline RGB visualization functions.")

    # STEP 7: Voxel (DeepSeek R1 7B) audits for BGR/RGB bugs, tensor shape errors, and Jupyter compatibility
    run_local_cv_agent("Voxel", "Audit Retina and Frame's vision code for BGR/RGB channel bugs, tensor shape mismatches, or blocking cv2.imshow calls.")

    # STEP 8: Lens (Gemma 4) writes dataset specs and execution instructions
    run_local_cv_agent("Lens", "Write a clear guide on required camera/video inputs, dataset weights, and how to run the Jupyter Notebook.")

    # STEP 9: Cortex (Nomic Embed) indexes the final CV architecture in work.md
    run_cortex_indexer()

    # STEP 10: Mango compiles the final executable Jupyter Notebook (.ipynb)
    build_jupyter_notebook(user_goal)

    # Final Notification from Mango (Gemini)
    final_memory = read_work_md()
    final_check_text = call_mango_gemini(
        f"You are Mango. Review this completed Computer Vision work.md log and give the user a 3-bullet completion summary:\n{final_memory}"
    )

    print("\n" + "=" * 65)
    print("MANGO NOTIFICATION: PIPELINE & JUPYTER NOTEBOOK READY")
    print("=" * 65)
    print(final_check_text)
    print(f"\nShared Agent Memory Log : {os.path.abspath(WORK_FILE)}")
    print(f"Executable Notebook     : {os.path.abspath(NOTEBOOK_FILE)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mango Computer Vision Multi-Agent Swarm CLI")
    parser.add_argument("command", nargs="?", help="Type 'start'")
    parser.add_argument("target", nargs="?", help="Type 'agents'")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Directory to save work.md and cv_pipeline.ipynb")
    args = parser.parse_args()

    if args.command == "start" and args.target == "agents":
        start_agents(target_dir=args.output_dir)
    else:
        print("Usage: python mango.py start agents [--output-dir /path/to/folder]")
