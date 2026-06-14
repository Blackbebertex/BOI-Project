import os
import re
import base64
import json
import zlib
import tempfile
import subprocess
import urllib.request
import urllib.error

# Define target paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DOC_PATH = os.path.join(PROJECT_ROOT, 'ENTERPRISE_TECHNICAL_DOCUMENTATION.md')
DIAGRAMS_DIR = os.path.join(PROJECT_ROOT, 'docs', 'diagrams')

# Ensure directories exist
os.makedirs(DIAGRAMS_DIR, exist_ok=True)

# List of all diagrams in sequential order of their occurrence in the document
DIAGRAM_INFOS = [
    {"filename": "6_1_batch_scoring_flow.png", "title": "Batch Scoring Flowchart"},
    {"filename": "6_2_single_account_scoring_flow.png", "title": "Single Account Scoring Flowchart"},
    {"filename": "6_3_decision_policy_change_flow.png", "title": "Decision Policy Change Flowchart"},
    {"filename": "6_4_manual_override_flow.png", "title": "Manual Override Flowchart"},
    {"filename": "6_5_api_request_lifecycle.png", "title": "API Request Lifecycle Flowchart"},
    {"filename": "6_6_model_startup_loading.png", "title": "Model Startup & Artifact Loading Flowchart"},
    {"filename": "6_7_1_batch_csv_upload.png", "title": "Batch CSV Upload Sequence Diagram"},
    {"filename": "6_7_2_shap_explanation.png", "title": "SHAP Explanation Request Sequence Diagram"},
    {"filename": "6_7_3_decision_override.png", "title": "Decision Override Sequence Diagram"},
    {"filename": "6_7_4_system_health_check.png", "title": "System Health Check Sequence Diagram"},
    {"filename": "7_1_c4_system_context.png", "title": "C4 System Context Diagram"},
    {"filename": "7_2_c4_container.png", "title": "C4 Container Diagram"},
    {"filename": "7_4_deployment_diagram.png", "title": "Deployment Diagram"},
    {"filename": "8_2_er_diagram.png", "title": "ER Diagram"}
]

# Standalone diagrams (not extracted from ENTERPRISE doc — used for hackathon proposal)
STANDALONE_DIAGRAMS = [
    {
        "filename": "9_1_scoring_typology_suspicion.png",
        "title": "End-to-End Scoring Pipeline with Typology and Suspicion Levels",
        "mermaid": """flowchart TD
    bankKeys["18 Bank Key Features"] --> fePipe["FeaturePipeline"]
    fePipe --> supervised["best_model.pkl (XGBoost)"]
    fePipe --> isoForest["Isolation Forest"]
    supervised --> fused["Fused Score (70% supervised + 30% anomaly)"]
    isoForest --> fused
    bankKeys --> typology["TypologyEngine"]
    typology --> boost["Typology Boost (max +0.10)"]
    fused --> adjusted["adjusted_fused_risk_score"]
    boost --> adjusted
    adjusted --> suspicion["suspicion_level (1-4)"]
    adjusted --> decision["Decision (BLOCK to APPROVE)"]
    typology --> flags["typology_flags"]""",
    },
]

def pako_deflate(data: bytes) -> bytes:
    """Compresses data using zlib with settings compatible with pako/mermaid."""
    compress = zlib.compressobj(9, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY)
    compressed_data = compress.compress(data)
    compressed_data += compress.flush()
    return compressed_data

def generate_mermaid_link(graph_markdown: str):
    """Generates a mermaid.ink URL for the given mermaid markdown."""
    j_graph = {
        "code": graph_markdown,
        "mermaid": {"theme": "default"}
    }
    byte_str = json.dumps(j_graph).encode('utf-8')
    deflated = pako_deflate(byte_str)
    b64_encoded = base64.b64encode(deflated).decode('ascii')
    url_safe_encoded = b64_encoded.replace('+', '-').replace('/', '_')
    return f"https://mermaid.ink/img/pako:{url_safe_encoded}"

def render_locally(graph_markdown: str, output_path: str) -> bool:
    """Tries to render the diagram locally using npx @mermaid-js/mermaid-cli."""
    # Create a temp file for the mermaid code
    with tempfile.NamedTemporaryFile(suffix='.mmd', delete=False, mode='w', encoding='utf-8') as f:
        f.write(graph_markdown)
        temp_mmd_name = f.name
    
    try:
        cmd = f'npx -y @mermaid-js/mermaid-cli -i "{temp_mmd_name}" -o "{output_path}"'
        print(f"  Trying local render: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=45)
        if result.returncode == 0:
            print(f"  [SUCCESS] Local render successful!")
            return True
        else:
            print(f"  [FAILED] Local render failed (code {result.returncode})")
            return False
    except Exception as e:
        print(f"  [ERROR] Local render failed with exception: {e}")
        return False
    finally:
        try:
            os.unlink(temp_mmd_name)
        except Exception:
            pass

def render_cloud(graph_markdown: str, output_path: str) -> bool:
    """Tries to render the diagram using mermaid.ink public API."""
    url = generate_mermaid_link(graph_markdown)
    print(f"  Trying cloud render via: {url[:100]}...")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            img_data = response.read()
        
        with open(output_path, 'wb') as f:
            f.write(img_data)
        print(f"  [SUCCESS] Cloud render successful!")
        return True
    except urllib.error.URLError as e:
        print(f"  [FAILED] Cloud render failed: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] Cloud render failed with exception: {e}")
        return False

def render_diagram(code: str, output_path: str, title: str) -> bool:
    """Render a single mermaid diagram to PNG (local first, cloud fallback)."""
    print(f"  Rendering '{title}' -> {os.path.basename(output_path)}...")
    success = render_locally(code, output_path)
    if not success:
        print("  Falling back to cloud rendering...")
        success = render_cloud(code, output_path)
    if not success:
        print(f"  [CRITICAL] Failed to render diagram '{title}'!")
    return success

def render_standalone_diagrams():
    """Render diagrams defined in STANDALONE_DIAGRAMS (e.g. hackathon proposal Fig 3)."""
    print(f"\nRendering {len(STANDALONE_DIAGRAMS)} standalone diagram(s)...")
    for info in STANDALONE_DIAGRAMS:
        output_path = os.path.join(DIAGRAMS_DIR, info["filename"])
        render_diagram(info["mermaid"], output_path, info["title"])
    print("Standalone diagram rendering complete.")

def process_diagrams():
    if not os.path.exists(DOC_PATH):
        print(f"Error: Documentation file not found at {DOC_PATH}")
        return

    print(f"Reading documentation from {DOC_PATH}...")
    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract all mermaid blocks
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
    print(f"Found {len(mermaid_blocks)} Mermaid diagrams in the document.")

    if len(mermaid_blocks) != len(DIAGRAM_INFOS):
        print(f"Error: Expected {len(DIAGRAM_INFOS)} diagrams, but found {len(mermaid_blocks)}.")
        return

    # Split documentation by mermaid blocks
    parts = re.split(r'```mermaid\n.*?\n```', content, flags=re.DOTALL)
    if len(parts) != len(DIAGRAM_INFOS) + 1:
        print(f"Error: Splitting content yielded {len(parts)} parts, expected {len(DIAGRAM_INFOS) + 1}.")
        return

    # Render each diagram and build updated content
    new_content = ""
    for i in range(len(DIAGRAM_INFOS)):
        new_content += parts[i]
        code = mermaid_blocks[i]
        info = DIAGRAM_INFOS[i]
        filename = info["filename"]
        title = info["title"]
        output_path = os.path.join(DIAGRAMS_DIR, filename)

        print(f"\n[{i+1}/{len(DIAGRAM_INFOS)}] Rendering '{title}' -> {filename}...")
        
        # Try local render first
        success = render_locally(code, output_path)
        
        # Fallback to cloud render if local fails
        if not success:
            print("  Falling back to cloud rendering...")
            success = render_cloud(code, output_path)

        if not success:
            print(f"  [CRITICAL] Failed to render diagram '{title}'!")
        
        # Construct updated markdown block with image and collapsible details
        # Note: We use relative path from the root doc location to the image
        rel_image_path = f"docs/diagrams/{filename}"
        replacement = (
            f"![{title}]({rel_image_path})\n\n"
            f"<details>\n"
            f"<summary>Show Mermaid Source</summary>\n\n"
            f"```mermaid\n"
            f"{code}\n"
            f"```\n"
            f"</details>"
        )
        new_content += replacement

    new_content += parts[-1]

    # Write updated documentation back
    print(f"\nWriting updated documentation to {DOC_PATH}...")
    with open(DOC_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Done! All diagrams processed successfully.")

if __name__ == "__main__":
    render_standalone_diagrams()
    process_diagrams()
