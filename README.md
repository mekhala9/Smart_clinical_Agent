# Smart Clinical Summary Agent
### GPT-4o-mini + ChromaDB RAG + OpenFDA + Symptom Intelligence
# Resources I used to learn and build this:
#    - Python basics:          https://www.w3schools.com/python/
#    - pandas (CSV reading):   https://pandas.pydata.org/docs/getting_started/intro_tutorials/
#    - OpenAI embeddings:      https://platform.openai.com/docs/guides/embeddings
#    - ChromaDB, RAG           Guide to RAG_DataBricks 
#    - What is ETL:            https://www.ibm.com/topics/etl
#    - What is RAG:            https://www.youtube.com/watch?v=T-D1OfcDW1M
#    - tqdm progress bar:      https://tqdm.github.io/
#    - hashlib explained:      https://docs.python.org/3/library/hashlib.html
#    - OpenAI chat API:        https://platform.openai.com/docs/guides/text-generation
#    - OpenFDA API docs:       https://open.fda.gov/apis/drug/label/
#    - requests library:       https://requests.readthedocs.io/en/latest/
#    - Python json module:     https://www.w3schools.com/python/python_json.asp
# ============================================================
Run it like this:   
cd C:\Users\mekha\OneDrive\Desktop\Smart_clinical_agent\clinical_agent
.\venv\Scripts\activate $env:OPENAI_API_KEY="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 "
python etl_pipeline.py
python clinical_summary_agent.py YOUR_PATIENT_ID

## Folder Structure
clinical_agent/
│
├── data/                      
│   ├── patients.csv
│   ├── encounters.csv
│   ├── conditions.csv
│   ├── medications.csv
│   └── observations.csv
│
├── chroma_db/    --Local vector database auto created during ETL pipeline run              
│
├── etl_pipeline.py                 ← STEP 1: Run once to build vector DB
├── clinical_summary_agent.py       ← STEP 2: Run for each patient
│
├── requirements.txt
├── setup_windows.bat               ← Windows quick setup
└── README.md

## Architecture

CSVs (5 files)
    │
[ETL PIPELINE]
  Extract  → Load & validate all 5 CSVs
  Transform → Chunk each patient into sections
  Load     → Embed via OpenAI → store in ChromaDB
    │
ChromaDB Vector Store
    │
    │  (at query time)
[AGENT — per Patient ID]
  ├── RAG Retrieve     → semantic search per section
  ├── Patient Details  → from patients.csv
  ├── Conditions Count → active (STOP = null)
  ├── Encounter Count  → from encounters.csv
  ├── Risk Score       → computed from encounter count
  ├── New Risk Score   → risk + active conditions
  ├── OpenFDA API      → drug purpose / warnings / side effects
  ├── Symptom Checker  → GPT-4o-mini analyses symptoms from CSV data
  └── GPT-4o-mini      → RAG context + structured data → summary
         │
        
    JSON Output (8 sections)

## Quick Start

### Step 1 — Open in VS Code
File → Open Folder → select clinical_agent/

### Step 2 — Set API Key
set OPENAI_API_KEY=sk-your-key-here

### Step 3 — Add Your CSV Files
Copy all 5 CSV files into the `data/` folder:

data/patients.csv
data/encounters.csv
data/conditions.csv
data/medications.csv
data/observations.csv

### Step 4 — Run ETL Pipeline (once)
python etl_pipeline.py

Expected output:
STAGE 1 — EXTRACT
 patients.csv    →  1,000 rows
 encounters.csv  →  8,432 rows
  ...
STAGE 2 — TRANSFORM
 Total chunks: 24,500
STAGE 3 — LOAD
 Total in ChromaDB: 24,500
 ETL COMPLETE — 142s

### Step 5 — Run the Agent
python clinical_summary_agent.py b9c610cd-28a6-4636-ccb6-c7a0d2a4cb85

## Output (JSON)

json
{
  "patient_id":               "b9c610cd-...",
  "full_name":                "Sai Mekhala Pondala",
  "dob":                      "12-09-1998",
  "gender":                   "Female",

  "active_conditions_count":  12,
  "active_conditions_list":   ["Hypertension", "Type 2 Diabetes", "..."],

  "encounter_count":          5,
  "risk_score":               5,
  "new_risk_score":           17,

  "medications": [
    {
      "drug":              "Metformin 500 MG",
      "purpose":           "For blood sugar control in type 2 diabetes...",
      "warnings":          "Risk of lactic acidosis...",
      "adverse_reactions": "Nausea, diarrhea, stomach upset..."
    }
  ],

  "symptoms_found": ["fever", "headache", "fatigue"],

  "clinical_summary": "A 27-year-old female presents with..."
}
Output is also saved as `summary_<patient_id>.json`

## Risk Score Logic
This is based upon encounters_count from encounters.csv file
When encounters_counts is between 1 - 3 then 2
When encounters_counts is between 4 - 9 then 5
When encounters_counts is between 10+ then 9

New Risk Score = risk_score + active_conditions_count


## API Keys Reference
| `OPENAI_API_KEY` | platform.openai.com |

> OpenFDA requires **no key** at all.

## Troubleshooting
| `Missing file: data/patients.csv` | Place all 5 CSVs in the `data/` folder |
| `ChromaDB not found` | Run `python etl_pipeline.py` first |
| `OPENAI_API_KEY not set` | Set the env variable (Step 3 above) |
| `Patient not found` | Check the exact patient ID in your CSV |
| OpenFDA returns no data | Drug name may differ — check `medications.csv` |
