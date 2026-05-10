# 🏥 Smart Clinical Summary Agent
### GPT-4o-mini + ChromaDB RAG + OpenFDA + Symptom Intelligence

---

## 📁 Folder Structure

```
clinical_agent/
│
├── 📂 data/                        ← PUT YOUR 5 CSV FILES HERE
│   ├── patients.csv
│   ├── encounters.csv
│   ├── conditions.csv
│   ├── medications.csv
│   └── observations.csv
│
├── 📂 chroma_db/                   ← Auto-created by ETL (don't touch)
│
├── etl_pipeline.py                 ← STEP 1: Run once to build vector DB
├── clinical_summary_agent.py       ← STEP 2: Run for each patient
│
├── requirements.txt
├── setup_windows.bat               ← Windows quick setup
├── setup_mac_linux.sh              ← Mac/Linux quick setup
└── README.md
```

---

## ⚙️ Architecture

```
CSVs (5 files)
    │
    ▼
[ETL PIPELINE]
  Extract   → Load & validate all 5 CSVs
  Transform → Chunk each patient into sections
  Load      → Embed via OpenAI → store in ChromaDB
    │
    ▼
[ChromaDB Vector Store]  ←──── persisted to disk
    │
    │  (at query time)
    ▼
[AGENT — per Patient ID]
  ├── RAG Retrieve     → semantic search per section
  ├── Patient Details  → from patients.csv
  ├── Conditions Count → active (STOP = null)
  ├── Encounter Count  → from encounters.csv
  ├── Risk Score       → computed from encounter count
  ├── New Risk Score   → risk + active conditions
  ├── OpenFDA API      → drug purpose / warnings / side effects
  ├── Symptom Checker  → GPT-4o-mini analyses symptoms from CSV data
  └── GPT-4o-mini      → RAG context + all data → clinical summary
         │
         ▼
    JSON Output (8 sections)
```

---

## 🚀 Quick Start

### Step 1 — Open in VS Code
```
File → Open Folder → select clinical_agent/
```

### Step 2 — Run Setup Script

**Windows** — in the VS Code terminal:
```powershell
.\setup_windows.bat
```

**Mac / Linux** — in the VS Code terminal:
```bash
chmod +x setup_mac_linux.sh
./setup_mac_linux.sh
```

### Step 3 — Set API Key

**Windows (VS Code terminal):**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

**Mac/Linux (VS Code terminal):**
```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Step 4 — Add Your CSV Files
Copy all 5 CSV files into the `data/` folder:
```
data/patients.csv
data/encounters.csv
data/conditions.csv
data/medications.csv
data/observations.csv
```

### Step 5 — Run ETL Pipeline (once only)
```powershell
python etl_pipeline.py
```

### Step 6 — Run the Agent
```powershell
python clinical_summary_agent.py YOUR_PATIENT_ID
```

---

## 📤 Output (JSON)

```json
{
  "patient_id":               "9acc871f-...",
  "full_name":                "Lavinia Heaney",
  "dob":                      "24-03-1985",
  "gender":                   "Female",

  "active_conditions_count":  5,
  "active_conditions_list":   ["Hypertension", "..."],

  "encounter_count":          5,
  "risk_score":               5,
  "new_risk_score":           10,

  "medications": [
    {
      "drug":              "Hydrochlorothiazide 25 MG Oral Tablet",
      "purpose":           "Hypertension",
      "warnings":          "...",
      "adverse_reactions": "..."
    }
  ],

  "symptoms_found": ["Hypertension", "Pain severity: 3/10"],

  "symptom_intelligence": {
    "possible_causes": [
      {"disease": "Essential Hypertension", "likelihood": "45%", "emergency": false}
    ],
    "severity":        "MODERATE — Hypertension with social stressors present",
    "triage_guidance": "Schedule follow-up with cardiologist within 2 weeks",
    "source":          "gpt-4o-mini"
  },

  "clinical_summary": "A 40-year-old female presents with..."
}
```
Output is also saved as `summary_<patient_id>.json`

---

## 📊 Risk Score Logic

| Encounter Count | Risk Score |
|---|---|
| 0 | 0 |
| 1 – 3 | 2 |
| 4 – 9 | 5 |
| 10+ | 9 |

**New Risk Score** = `risk_score` + `active_conditions_count`

---

## 🔑 API Keys

| Key | Required | Where to get |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes — only key needed | platform.openai.com |

> OpenFDA requires **no key** at all — completely free.
> Symptom checking uses GPT-4o-mini — no extra key needed.

---

## ❓ Troubleshooting

| Error | Fix |
|---|---|
| `Missing file: data/patients.csv` | Place all 5 CSVs in the `data/` folder |
| `ChromaDB not found` | Run `python etl_pipeline.py` first |
| `OPENAI_API_KEY not set` | Set the env variable (Step 3 above) |
| `Patient not found` | Check the exact patient ID in your CSV |
| `Connection error` during ETL | Check your internet — it will auto-retry |
| OpenFDA returns no data | Drug name may differ — check `medications.csv` |
