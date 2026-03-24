 import os
import re
import streamlit as st
import streamlit.components.v1 as components
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

import zipfile
import requests
from pathlib import Path

from dotenv import load_dotenv

load_dotenv() # Load variables from .env if present

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
CHROMA_ZIP_URL = "https://storage.googleapis.com/scholar-bucket-n/chroma_db.zip" #GCS URL

def download_and_unzip_db():
    """Downloads the ChromaDB zip if it doesn't exist and unzips it."""
    if not os.path.exists(DB_DIR):
        st.info("📦 Database not found. Initializing download...")
        try:
            zip_path = os.path.join(BASE_DIR, "chroma_db.zip")
            if not os.path.exists(zip_path):
                response = requests.get(CHROMA_ZIP_URL, stream=True)
                if response.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    st.error(f"❌ Failed to download database. Status code: {response.status_code}")
                    return False

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)
            os.remove(zip_path) # Clean up
            st.success("✅ Database initialized successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error initializing database: {e}")
            return False
    return True

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Chroma Collection ──────────────────────────────────────────────────────────
@st.cache_resource
def get_chroma_collection():
    if not os.path.exists(DB_DIR):
        if "YOUR_DIRECT_DOWNLOAD_LINK_HERE" in CHROMA_ZIP_URL:
            st.error("⚠️ Database missing! Please update `CHROMA_ZIP_URL` in `app.py` or manually upload `chroma_db/`.")
            st.stop()
        if not download_and_unzip_db():
            st.stop()
            
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
    )
    return chroma_client.get_collection(name="exam_syllabus_collection", embedding_function=openai_ef)

# ── Exam Generation ────────────────────────────────────────────────────────────
def generate_exam(level, subject, term, question_count, exam_format, difficulty,
                  logic_level, include_diagrams, image_model, custom_image_prompt,
                  ai_model, temperature, top_p):
    collection = get_chroma_collection()
    search_query = f"{level} {subject} {term} exam questions notes syllabus"
    where_filter = {
        "$and": [
            {"level": {"$in": [level, f"P.{level[-1]}"] if level.startswith("P") else [level]}},
            {"subject": {"$eq": subject}}
        ]
    }
    try:
        results = collection.query(query_texts=[search_query], n_results=15, where=where_filter)
    except Exception as e:
        return f"Database query failed: {e}", f"Database query failed: {e}"

    if not results["documents"] or not results["documents"][0]:
        msg = f"No '{subject}' content found for '{level}'."
        return msg, msg

    retrieved_context = "\n\n---\n\n".join(results["documents"][0])

    diagram_instruction = ""
    if include_diagrams:
        diagram_instruction = (
            "\n7. DIAGRAMS: Where a question needs a visual, insert: [GENERATE_IMAGE: <description>]"
            "\n   Place it directly below that question. Aim for 1-2 diagram placeholders if applicable."
        )

    system_prompt = f"""You are an expert Ugandan Primary School teacher generating a professional exam.

EXAM DETAILS: Level={level}, Subject={subject}, Term={term}, Format={exam_format},
Questions={question_count}, Difficulty={difficulty}, Logic Level={logic_level}

INSTRUCTIONS:
1. Use ONLY the SYLLABUS CONTEXT below.
2. DO NOT include a title/name/school header — those are on the cover page. Start at SECTION A.
3. Number all questions clearly.
4. After EVERY question add dotted answer lines:
   Short answer (1 line):  ...................................................................................................................
   Medium (2 lines): two such rows. Long/essay (4+ lines): four rows.
   Multiple choice: list A B C D cleanly — no dots needed.
5. Adjust to difficulty ({difficulty}) and logic level ({logic_level}).
6. End with a MARKING GUIDE / ANSWER KEY section.{diagram_instruction}

SYLLABUS CONTEXT:
{retrieved_context}
"""
    response = client.chat.completions.create(
        model=ai_model,
        messages=[{"role": "system", "content": system_prompt}],
        temperature=temperature,
        top_p=top_p,
    )
    exam_content = response.choices[0].message.content

    if include_diagrams:
        style_prompt = (
            " Avoid unnecessary decorations; exam-ready diagram."
            " Style: Educational infographic, vector-style geometry, clean labels."
            " Use minimal colors, soft stroke outlines. NO TEXT."
        )
        for match in re.finditer(r"\[GENERATE_IMAGE:\s*(.*?)\]", exam_content):
            full_match = match.group(0)
            img_prompt = match.group(1)
            try:
                enhanced = img_prompt + style_prompt
                if custom_image_prompt and custom_image_prompt.strip():
                    enhanced += " ADDITIONAL: " + custom_image_prompt
                img_url = ""
                if "DALL-E 3" in image_model:
                    r = client.images.generate(model="dall-e-3", prompt=enhanced, size="1024x1024", quality="standard", n=1)
                    img_url = r.data[0].url
                elif "Gemini" in image_model:
                    import requests as req
                    gkey = "AIzaSyBLcNvUXNbVKOVIufeZx1e0zkv_6ONV9-E"
                    resp = req.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict?key={gkey}",
                        json={"instances": [{"prompt": enhanced}], "parameters": {"sampleCount": 1}},
                    )
                    if resp.status_code == 200:
                        b64 = resp.json().get("predictions", [{}])[0].get("bytesBase64Encoded", "")
                        img_url = f"data:image/jpeg;base64,{b64}" if b64 else ""
                    else:
                        raise Exception(f"Gemini error {resp.status_code}")
                styled = f'<div style="text-align:center;margin:24px 0;"><img src="{img_url}" style="max-width:70%;border:1px solid #ccc;"/></div>'
                exam_content = exam_content.replace(full_match, styled)
            except Exception as e:
                exam_content = exam_content.replace(full_match, f"<i>(Diagram failed: {e})</i>")

    exam_html = exam_content.replace("\n", "<br>")
    return exam_content, exam_html


# ── HTML Builder with Live Style Editor ───────────────────────────────────────
def build_full_html(level, subject, term_roman, exam_year, duration, school_name,
                    brand_name, question_count, exam_html):

    exam_rows_html = ""
    i = 1
    while i <= question_count:
        end = min(i + 9, question_count)
        exam_rows_html += f"<tr><td>{i}-{end}</td><td></td><td></td></tr>"
        i += 10
    exam_rows_html += "<tr class='total-row'><td><strong>TOTAL</strong></td><td></td><td></td></tr>"

    school_block = f"<div class='school-display'>{school_name.strip()}</div>" if school_name.strip() else ""
    sa_q = round(question_count * 0.6)
    sb_q = round(question_count * 0.4)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
/* ── CSS VARIABLES ── */
:root {{
  --heading-color: #000;
  --heading-size: 22px;
  --border-color: #000;
  --border-width: 3px;
  --border-style: solid;
  --border-radius: 0px;
  --page-bg: #fff;
  --body-font-size: 15px;
  --body-line-height: 1.9;
  --footer-color: #555;
  --footer-size: 11px;
  --logo-size: 44px;
  --logo-color: #000000;
  --logo-spacing: 5px;
  --logo-weight: 900;
  --section-color: #000;
}}
/* ── PRINT ── */
@page {{ size: A4; margin: 18mm 15mm 20mm 15mm; }}
@media print {{
  body {{ background: white !important; padding: 0 !important; }}
  .toolbar, .style-panel {{ display: none !important; }}
  .a4-page {{ box-shadow: none !important; margin: 0 !important; page-break-after: always; }}
  .a4-page:last-child {{ page-break-after: avoid; }}
}}
/* ── BASE ── */
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: Arial, Helvetica, sans-serif;
  background: #d5d5d5;
  padding: 20px 10px 80px;
  color: #000;
}}
/* ── TOOLBAR ── */
.toolbar {{
  position: fixed; bottom: 0; left: 0; right: 0;
  background: #1e293b;
  display: flex; align-items: center; gap: 10px;
  padding: 10px 20px; z-index: 9999;
  box-shadow: 0 -4px 16px rgba(0,0,0,0.3);
}}
.toolbar-label {{
  color: #94a3b8; font-size: 12px; font-weight: 600;
  letter-spacing: 1px; text-transform: uppercase; margin-right: auto;
}}
.tbtn {{
  padding: 9px 20px; border: none; border-radius: 20px;
  cursor: pointer; font-size: 13px; font-weight: 700; transition: 0.2s;
}}
.tbtn-style {{ background: #7c3aed; color: white; }}
.tbtn-style:hover {{ background: #6d28d9; }}
.tbtn-print {{ background: #1a56db; color: white; }}
.tbtn-print:hover {{ background: #1e429f; }}
/* ── STYLE PANEL ── */
.style-panel {{
  position: fixed; top: 0; right: -380px; width: 360px; height: 100vh;
  background: #1e293b; color: #e2e8f0;
  z-index: 10000; box-shadow: -6px 0 30px rgba(0,0,0,0.4);
  transition: right 0.3s cubic-bezier(0.4,0,0.2,1);
  overflow-y: auto; font-family: Arial, sans-serif;
}}
.style-panel.open {{ right: 0; }}
.panel-header {{
  background: #0f172a; padding: 18px 20px; font-size: 16px; font-weight: 700;
  border-bottom: 1px solid #334155;
  display: flex; justify-content: space-between; align-items: center;
  position: sticky; top: 0; z-index: 1;
}}
.panel-close {{
  cursor: pointer; font-size: 20px; background: none; border: none;
  color: #94a3b8; line-height: 1;
}}
.panel-close:hover {{ color: #e2e8f0; }}
.psec {{ padding: 16px 20px; border-bottom: 1px solid #334155; }}
.psec-title {{
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1.5px; color: #7c3aed; margin-bottom: 12px;
}}
.crow {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }}
.clabel {{ font-size: 13px; color: #cbd5e1; }}
.cinput {{
  background: #0f172a; border: 1px solid #475569; border-radius: 6px;
  color: #e2e8f0; padding: 5px 8px; font-size: 13px; width: 136px;
}}
.cinput:focus {{ outline: none; border-color: #7c3aed; }}
input[type="color"] {{
  width: 48px; height: 32px; padding: 2px; border-radius: 6px;
  cursor: pointer; border: 1px solid #475569; background: #0f172a;
}}
input[type="range"] {{ width: 100px; accent-color: #7c3aed; }}
.rval {{ font-size: 12px; color: #94a3b8; min-width: 38px; text-align: right; }}
.preset-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
.preset-btn {{
  padding: 5px 10px; border: 1px solid #475569; border-radius: 6px;
  background: #0f172a; color: #e2e8f0; font-size: 12px; cursor: pointer;
}}
.preset-btn:hover {{ background: #334155; }}
.preset {{
  width: 100%; margin-top: 6px; background: #334155; color: #e2e8f0;
  border: none; border-radius: 8px; padding: 9px; cursor: pointer;
  font-size: 13px; font-weight: 600;
}}
.preset:hover {{ background: #475569; }}
/* ── A4 PAGE ── */
.a4-page {{
  width: 794px; min-height: 1123px;
  background: var(--page-bg); margin: 0 auto 24px auto;
  padding: 25mm 18mm 22mm 18mm;
  box-shadow: 0 4px 20px rgba(0,0,0,0.18);
  border: var(--border-width) var(--border-style) var(--border-color);
  border-radius: var(--border-radius); position: relative;
}}
.pgnum {{
  position: absolute; bottom: 10mm; left: 18mm;
  width: calc(100% - 36mm); text-align: center;
  font-size: var(--footer-size); color: var(--footer-color);
}}
/* ── COVER ── */
.logo-block {{ text-align: center; margin-bottom: 12px; }}
.logo-pill {{
  display: inline-block;
  padding: 4px 36px;
  font-size: var(--logo-size);
  font-weight: var(--logo-weight);
  color: var(--logo-color);
  letter-spacing: var(--logo-spacing);
  line-height: 1.2;
}}
.logo-icons {{ font-size: 28px; margin: 6px 0 2px; }}
.logo-sub {{ font-size: 13px; font-weight: 700; letter-spacing: 4px; text-transform: uppercase; }}
.school-display {{ text-align: center; font-size: 15px; font-weight: 600; margin: 6px 0 0; }}
.divider {{ border-top: 2px solid #000; margin: 14px 0 10px; }}
.exam-main-title {{
  text-align: center; font-size: var(--heading-size);
  font-weight: 900; text-transform: uppercase;
  margin-bottom: 8px; color: var(--heading-color);
}}
.exam-sub-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }}
.exam-subject-label {{ font-size: 19px; font-weight: 900; text-transform: uppercase; color: var(--section-color); }}
.exam-time-label {{ font-size: 15px; font-weight: 900; text-transform: uppercase; }}
.name-school-box {{ border: 1px solid #000; padding: 10px 14px; margin-bottom: 14px; }}
.ns-row {{ display: flex; align-items: flex-end; margin-bottom: 8px; }}
.ns-row:last-child {{ margin-bottom: 0; }}
.ns-label {{ font-weight: 700; font-size: 14px; width: 70px; flex-shrink: 0; }}
.ns-colon {{ margin: 0 6px 3px; font-weight: 700; }}
.ns-line {{ flex-grow: 1; border-bottom: 1.5px solid #000; height: 18px; }}
.instr-header {{ font-size: 13px; font-weight: 900; text-transform: uppercase; text-decoration: underline; margin: 14px 0 10px; }}
.two-col {{ display: flex; gap: 24px; align-items: flex-start; }}
.instructions-col {{ flex: 1.1; font-size: 13px; line-height: 1.8; }}
.instructions-col li {{ list-style: none; padding-left: 0; margin-bottom: 5px; }}
.instructions-col li::before {{ content: "o  "; }}
.examiner-col {{ flex: 0.85; }}
.time-box {{ border: 1px solid #000; padding: 5px 10px; font-size: 13px; font-weight: 700; margin-bottom: 8px; }}
.examiner-title {{ font-weight: 900; font-size: 13px; text-align: center; margin-bottom: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ border: 1px solid #000; padding: 5px 8px; font-weight: 700; text-align: left; background: #f5f5f5; }}
td {{ border: 1px solid #000; padding: 5px 8px; height: 24px; }}
.total-row td {{ font-weight: 900; }}
/* ── EXAM BODY ── */
.body {{ font-size: var(--body-font-size); line-height: var(--body-line-height); }}
.body h1 {{ font-size: 17px; font-weight: 900; text-transform: uppercase; text-align: center; border-bottom: 2px solid var(--section-color); padding-bottom: 6px; margin: 24px 0 12px; color: var(--section-color); }}
.body h2 {{ font-size: 15px; font-weight: 900; text-transform: uppercase; margin: 20px 0 8px; color: var(--section-color); }}
.body h3 {{ font-size: 14px; font-weight: 700; margin: 16px 0 6px; }}
.body p {{ margin-bottom: 14px; }}
.body strong {{ font-weight: 900; }}
</style>
</head>
<body>

<!-- ── STYLE PANEL ── -->
<div class="style-panel" id="stylePanel">
  <div class="panel-header">
    🎨 Live Style Editor
    <button class="panel-close" onclick="togglePanel()">✕</button>
  </div>

  <div class="psec">
    <div class="psec-title">🎨 Colour Presets</div>
    <div class="preset-row">
      <button class="preset-btn" onclick="applyPreset('classic')">Classic Black</button>
      <button class="preset-btn" onclick="applyPreset('blue')">Royal Blue</button>
      <button class="preset-btn" onclick="applyPreset('green')">Forest Green</button>
      <button class="preset-btn" onclick="applyPreset('maroon')">Maroon</button>
    </div>
  </div>

  <div class="psec">
    <div class="psec-title">📌 Heading Title</div>
    <div class="crow">
      <span class="clabel">Text</span>
      <input class="cinput" type="text" id="headingText"
        value="Beginning of Term {term_roman} Examination {exam_year}"
        oninput="document.getElementById('mainTitle').textContent=this.value">
    </div>
    <div class="crow">
      <span class="clabel">Color</span>
      <input type="color" value="#000000" oninput="sv('--heading-color',this.value)">
    </div>
    <div class="crow">
      <span class="clabel">Font Size</span>
      <input type="range" min="14" max="36" value="22"
        oninput="sv('--heading-size',this.value+'px');this.nextElementSibling.textContent=this.value+'px'">
      <span class="rval">22px</span>
    </div>
  </div>

  <div class="psec">
    <div class="psec-title">🏷️ Subject / Section Label</div>
    <div class="crow">
      <span class="clabel">Color</span>
      <input type="color" value="#000000" oninput="sv('--section-color',this.value)">
    </div>
  </div>

  <div class="psec">
    <div class="psec-title">🔤 Brand / Logo Text</div>
    <div class="crow">
      <span class="clabel">Logo Words</span>
      <input class="cinput" type="text" id="logoText"
        value="{brand_name}"
        oninput="document.getElementById('logoPill').textContent=this.value">
    </div>
    <div class="crow">
      <span class="clabel">Color</span>
      <input type="color" value="#000000" oninput="sv('--logo-color',this.value)">
    </div>
    <div class="crow">
      <span class="clabel">Font Size</span>
      <input type="range" min="24" max="64" value="44"
        oninput="sv('--logo-size',this.value+'px');this.nextElementSibling.textContent=this.value+'px'">
      <span class="rval">44px</span>
    </div>
    <div class="crow">
      <span class="clabel">Letter Spacing</span>
      <input type="range" min="0" max="20" value="5"
        oninput="sv('--logo-spacing',this.value+'px');this.nextElementSibling.textContent=this.value+'px'">
      <span class="rval">5px</span>
    </div>
    <div class="crow">
      <span class="clabel">Font Weight</span>
      <select class="cinput" onchange="sv('--logo-weight',this.value)">
        <option value="400">Normal (400)</option>
        <option value="600">Medium (600)</option>
        <option value="700">Bold (700)</option>
        <option value="900" selected>Extra Bold (900)</option>
      </select>
    </div>
  </div>

  <div class="psec">
    <div class="psec-title">🔲 Page Border</div>
    <div class="crow">
      <span class="clabel">Color</span>
      <input type="color" value="#000000" oninput="sv('--border-color',this.value)">
    </div>
    <div class="crow">
      <span class="clabel">Width</span>
      <input type="range" min="0" max="12" value="3"
        oninput="sv('--border-width',this.value+'px');this.nextElementSibling.textContent=this.value+'px'">
      <span class="rval">3px</span>
    </div>
    <div class="crow">
      <span class="clabel">Style</span>
      <select class="cinput" onchange="sv('--border-style',this.value)">
        <option>solid</option><option>dashed</option>
        <option>dotted</option><option>double</option><option>none</option>
      </select>
    </div>
    <div class="crow">
      <span class="clabel">Rounded</span>
      <input type="range" min="0" max="32" value="0"
        oninput="sv('--border-radius',this.value+'px');this.nextElementSibling.textContent=this.value+'px'">
      <span class="rval">0px</span>
    </div>
  </div>

  <div class="psec">
    <div class="psec-title">📄 Page Background</div>
    <div class="crow">
      <span class="clabel">Color</span>
      <input type="color" value="#ffffff" oninput="sv('--page-bg',this.value)">
    </div>
  </div>

  <div class="psec">
    <div class="psec-title">📝 Body Text</div>
    <div class="crow">
      <span class="clabel">Font Size</span>
      <input type="range" min="11" max="20" value="15"
        oninput="sv('--body-font-size',this.value+'px');this.nextElementSibling.textContent=this.value+'px'">
      <span class="rval">15px</span>
    </div>
    <div class="crow">
      <span class="clabel">Line Spacing</span>
      <input type="range" min="12" max="30" value="19"
        oninput="sv('--body-line-height',(this.value/10).toFixed(1));this.nextElementSibling.textContent=(this.value/10).toFixed(1)">
      <span class="rval">1.9</span>
    </div>
  </div>

  <div class="psec">
    <div class="psec-title">📋 Footer</div>
    <div class="crow">
      <span class="clabel">Custom Text</span>
      <input class="cinput" type="text" placeholder="e.g. EDUMERC © 2026" oninput="updateFooter(this.value)">
    </div>
    <div class="crow">
      <span class="clabel">Color</span>
      <input type="color" value="#555555" oninput="sv('--footer-color',this.value)">
    </div>
    <div class="crow">
      <span class="clabel">Size</span>
      <input type="range" min="8" max="16" value="11"
        oninput="sv('--footer-size',this.value+'px');this.nextElementSibling.textContent=this.value+'px'">
      <span class="rval">11px</span>
    </div>
  </div>

  <div class="psec">
    <button class="preset" onclick="resetAll()">↺ Reset to Defaults</button>
  </div>
</div>

<!-- ── BOTTOM TOOLBAR ── -->
<div class="toolbar">
  <span class="toolbar-label">✏️ Edumerc Exam Editor</span>
  <button class="tbtn tbtn-style" onclick="togglePanel()">🎨 Style Editor</button>
  <button class="tbtn tbtn-print" onclick="window.print()">📥 Download PDF</button>
</div>

<!-- ── PAGE 1: COVER ── -->
<div class="a4-page">
  <div class="logo-block">
    <div class="logo-pill" id="logoPill">{brand_name}</div>
    <div class="logo-sub">Examinations Services</div>
    {school_block}
  </div>
  <div class="divider"></div>
  <div class="exam-main-title" id="mainTitle">Beginning of Term {term_roman} Examination {exam_year}</div>
  <div class="exam-sub-row">
    <span class="exam-subject-label">{level} &nbsp;–&nbsp; {subject}</span>
    <span class="exam-time-label">TIME: {duration}</span>
  </div>
  <div class="name-school-box">
    <div class="ns-row"><span class="ns-label">NAME</span><span class="ns-colon">:</span><div class="ns-line"></div></div>
    <div class="ns-row"><span class="ns-label">SCHOOL</span><span class="ns-colon">:</span><div class="ns-line"></div></div>
  </div>
  <div class="instr-header">Read the following instructions carefully before opening</div>
  <div class="two-col">
    <div class="instructions-col">
      <ul>
        <li>This paper has two sections A and B.</li>
        <li>Section A has {sa_q} questions ({sa_q} marks).</li>
        <li>Section B has {sb_q} questions ({sb_q * 4} marks).</li>
        <li>Attempt all questions in both sections.</li>
        <li>All answers must be written in black or blue point pens or ink.</li>
        <li>Only diagrams must be drawn in pencil.</li>
        <li>Any hand writing that cannot be easily read may lead to loss of marks.</li>
        <li>Un necessary alteration of work will lead to loss of marks.</li>
        <li>Do not fill anything in the boxes indicated for examiner's use only.</li>
      </ul>
    </div>
    <div class="examiner-col">
      <div class="time-box">TIME &nbsp;:&nbsp; {duration}</div>
      <div class="examiner-title">For examiner's use only</div>
      <table>
        <tr><th>Question</th><th>Marks</th><th>EXR'S</th></tr>
        {exam_rows_html}
      </table>
    </div>
  </div>
  <div class="pgnum"><span class="ft">Page 1</span></div>
</div>

<!-- ── PAGE 2+: EXAM BODY ── -->
<div class="a4-page">
  <div class="body">{exam_html}</div>
  <div class="pgnum"><span class="ft">Page 2</span></div>
</div>

<script>
function sv(name, val) {{ document.documentElement.style.setProperty(name, val); }}

function togglePanel() {{ document.getElementById('stylePanel').classList.toggle('open'); }}

function updateFooter(text) {{
  document.querySelectorAll('.ft').forEach(function(el, i) {{
    el.textContent = text.trim() ? text + ' — Page ' + (i+1) : 'Page ' + (i+1);
  }});
}}

const presets = {{
  classic: {{ '--heading-color':'#000','--section-color':'#000','--border-color':'#000','--page-bg':'#fff' }},
  blue:    {{ '--heading-color':'#1a3a8f','--section-color':'#1a3a8f','--border-color':'#1a3a8f','--page-bg':'#f0f4ff' }},
  green:   {{ '--heading-color':'#1a5c2f','--section-color':'#1a5c2f','--border-color':'#1a5c2f','--page-bg':'#f0fff4' }},
  maroon:  {{ '--heading-color':'#6b0f1a','--section-color':'#6b0f1a','--border-color':'#6b0f1a','--page-bg':'#fff8f8' }},
}};

function applyPreset(name) {{
  const p = presets[name];
  if (!p) return;
  Object.entries(p).forEach(([k,v]) => sv(k,v));
}}

function resetAll() {{
  const defs = {{
    '--heading-color':'#000','--heading-size':'22px',
    '--border-color':'#000','--border-width':'3px','--border-style':'solid','--border-radius':'0px',
    '--page-bg':'#fff','--body-font-size':'15px','--body-line-height':'1.9',
    '--footer-color':'#555','--footer-size':'11px',
    '--logo-size':'44px','--logo-color':'#000000','--logo-spacing':'5px','--logo-weight':'900',
    '--section-color':'#000',
  }};
  Object.entries(defs).forEach(([k,v]) => sv(k,v));
}}
</script>
</body>
</html>"""


# ── Streamlit App ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Edumerc Exam Generator", page_icon="📝", layout="wide")
st.title("📝 Edumerc AI Exam Generator")
st.markdown("Generate curriculum-accurate exam papers from your school's notes, schemes & past papers.")

with st.sidebar:
    st.header("📋 Exam Details")
    brand_name     = st.text_input("Brand / Academy Name", value="EDUMERC")
    school_name    = st.text_input("School Name (optional)", placeholder="e.g. St. Mary's Primary")
    level          = st.selectbox("Level", ["P1","P2","P3","P4","P5","P6","P7","Nursery","Baby Class","Middle Class","Top Class"])
    subject        = st.selectbox("Subject", ["Mathematics","English","Science","Social Studies","Religious Education","Literacy"])
    term           = st.selectbox("Term", ["Term 1","Term 2","Term 3"])
    exam_year      = st.text_input("Year", value="2026")
    duration       = st.text_input("Duration", value="2 HR 30 MINUTES")
    question_count = st.slider("Number of Questions", 5, 50, 20)

    st.markdown("---")
    st.header("🧠 Engine Tuning")
    exam_format  = st.selectbox("Format", ["Mixed (Section A & B)","Multiple Choice Only","Short Answer","Structural/Essay"])
    difficulty   = st.selectbox("Difficulty", ["Standard","Easy (Remedial)","Hard (Advanced)","Expert (Challenge)"])
    logic_level  = st.selectbox("Logic Level", ["Balanced","Basic Recall","Application","Critical Thinking"])

    st.markdown("---")
    st.header("🤖 AI Model Settings")
    ai_model    = st.selectbox("AI Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, step=0.05, help="Higher = more creative/random, Lower = more focused/deterministic")
    top_p       = st.slider("Top P", 0.0, 1.0, 1.0, step=0.05, help="Alternative to temperature. Controls diversity of output.")

    st.markdown("---")
    st.header("🎨 Visual Engine")
    include_diagrams = st.checkbox("Auto-Generate Diagrams", False)
    image_model, custom_image_prompt = "DALL-E 3 (OpenAI)", ""
    if include_diagrams:
        image_model = st.selectbox("Model", ["DALL-E 3 (OpenAI)","Gemini / Imagen 4 (Google)","Nanobanana"])
        custom_image_prompt = st.text_area("Custom Diagram Notes", placeholder="e.g. Use dotted label lines only")

    st.markdown("---")
    generate_btn = st.button("🚀 Generate Exam Paper", use_container_width=True, type="primary")

if generate_btn:
    with st.spinner("Generating your exam paper..."):
        try:
            term_roman = {"Term 1":"I","Term 2":"II","Term 3":"III"}.get(term, term.upper())
            exam_raw, exam_html = generate_exam(
                level, subject, term, question_count, exam_format,
                difficulty, logic_level, include_diagrams, image_model, custom_image_prompt,
                ai_model, temperature, top_p
            )
            st.success("✅ Exam Generated!")
            st.markdown("---")

            full_html = build_full_html(
                level, subject, term_roman, exam_year, duration,
                school_name, brand_name, question_count, exam_html,
            )
            components.html(full_html, height=3200, scrolling=True)

            st.markdown("---")
            st.download_button(
                label="📥 Download Exam (Plain Text)",
                data=f"{brand_name} EXAMINATIONS SERVICES\nBEGINNING OF TERM {term_roman} EXAMINATION {exam_year}\n{level} - {subject}  |  TIME: {duration}\n\nNAME: _______________\nSCHOOL: _____________\n\n{exam_raw}",
                file_name=f"{brand_name}_{level}_{subject.replace(' ','_')}_{term}_{exam_year}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
