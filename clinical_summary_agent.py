# ============================================================
#  SMART CLINICAL SUMMARY AGENT
#  What this file does:
#    1. Takes a Patient ID as input
#    2. Retrieves relevant patient data from ChromaDB (RAG)
#    3. Computes risk scores and active conditions
#    4. Looks up drug info from OpenFDA API (free, no key needed)
#    5. Checks symptoms using GPT-4o-mini
#    6. Generates a clinical summary using GPT-4o-mini
#    7. Returns everything as a structured JSON output
#
#  Run it like this:
# $env:OPENAI_API_KEY="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"
#    python clinical_summary_agent.py YOUR_PATIENT_ID

import os
import sys
import json
import re
import requests
import pandas as pd
from datetime import datetime
from openai import OpenAI
import chromadb

#  SETTINGS

CSV_FOLDER      = "./data"
CHROMA_FOLDER   = "./chroma_db"
COLLECTION_NAME = "clinical_records"
GPT_MODEL       = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENFDA_URL     = "https://api.fda.gov/drug/label.json"

# How many chunks to retrieve from ChromaDB per section
CHUNKS_PER_SECTION = 3

# Read the API key you set in the terminal
# Windows:    $env:OPENAI_API_KEY="sk-..."
MY_API_KEY = os.environ.get("OPENAI_API_KEY", "")

#  SMALL HELPER FUNCTIONS
#  Reusable tools used throughout the file

def calculate_age(date_of_birth):
    # Tries different date formats because CSVs are inconsistent
    date_formats = ["%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%SZ"]
    for date_format in date_formats:
        try:
            birth_date = datetime.strptime(date_of_birth, date_format)
            age_in_days = (datetime.today() - birth_date).days
            return age_in_days // 365
        except ValueError:
            continue
    return 0  # if we can't figure it out, return 0


def find_id_column(table):
    # Find which column holds the patient ID in a given table
    # Different CSV files use different names for the same thing
    for col in table.columns:
        if col in ("ID", "PATIENT", "PATIENT_ID"):
            return col
    raise KeyError("Cannot find patient ID column. Columns found: " + str(table.columns.tolist()))


def find_column(table, *possible_names):
    # Try multiple possible column names and return the first match
    for name in possible_names:
        if name in table.columns:
            return name
    return None


def get_value(row, *possible_columns):
    # Get a value from a row, trying multiple column names
    # Returns "N/A" if nothing is found
    for col in possible_columns:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return "N/A"

#  LOADING DATA
#  Read all 5 CSV files and connect to ChromaDB

def load_csv_files():
    tables = {}
    file_names = ["patients", "encounters", "conditions", "medications", "observations"]

    for name in file_names:
        file_path = os.path.join(CSV_FOLDER, name + ".csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError("Missing file: " + file_path + "\nPlease put all 5 CSVs in the data/ folder")

        table = pd.read_csv(file_path, low_memory=False)
        table.columns = [col.strip().upper() for col in table.columns]
        tables[name] = table

    print("  All CSV files loaded")
    return tables


def connect_to_chromadb():
    if not os.path.exists(CHROMA_FOLDER):
        raise FileNotFoundError(
            "ChromaDB not found at " + CHROMA_FOLDER + "\n"
            "Please run etl_pipeline.py first!"
        )

    chroma_client = chromadb.PersistentClient(path=CHROMA_FOLDER)
    collection    = chroma_client.get_collection(COLLECTION_NAME)

    print("  ChromaDB connected — " + str(collection.count()) + " chunks available")
    return collection

#  RAG — RETRIEVE RELEVANT CHUNKS
#  Search ChromaDB for the most relevant text for this patient


# 2. API call - get embeddings for text ##################################

def get_embedding_for_text(openai_client, text):
    # Convert a piece of text into a vector (list of numbers)
    # We use this to search ChromaDB by meaning
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def search_chromadb(openai_client, collection, patient_id, search_query):
    # Convert our search query into a vector
    query_vector = get_embedding_for_text(openai_client, search_query)

    # Search ChromaDB for the most similar chunks
    # We filter by patient_id so we only get THIS patient's data
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=CHUNKS_PER_SECTION,
        where={"patient_id": patient_id},
        include=["documents"]
    )

    # Return the list of matching text chunks
    return results.get("documents", [[]])[0]


def retrieve_patient_context(openai_client, collection, patient_id):
    # Run 5 different searches — one for each section of patient data
    # This gives us the most relevant chunks across all sections
    search_queries = {
        "DEMOGRAPHICS": "patient name age gender date of birth " + patient_id,
        "CONDITIONS":   "active diagnoses medical conditions " + patient_id,
        "MEDICATIONS":  "current medications prescriptions drugs " + patient_id,
        "ENCOUNTERS":   "hospital visits clinical encounters " + patient_id,
        "OBSERVATIONS": "lab results vitals symptoms observations " + patient_id,
    }

    all_context_parts = []

    for section_name, query in search_queries.items():
        chunks = search_chromadb(openai_client, collection, patient_id, query)
        if chunks:
            section_text = "[" + section_name + "]\n" + "\n---\n".join(chunks)
            all_context_parts.append(section_text)

    # Join all sections into one big context string
    full_context = "\n\n".join(all_context_parts)
    return full_context

#  OUTPUT 1 — PATIENT DETAILS
#  Get basic info: name, DOB, gender

def get_patient_details(tables, patient_id):
    patients_table = tables["patients"]
    id_col         = find_id_column(patients_table)

    # Find the row for this patient
    matching_rows = patients_table[patients_table[id_col].astype(str).str.strip() == patient_id]

    if matching_rows.empty:
        raise ValueError("Patient ID not found: " + patient_id)

    row = matching_rows.iloc[0]

    first_name = get_value(row, "FIRST", "FIRSTNAME")
    last_name  = get_value(row, "LAST",  "LASTNAME")
    dob        = get_value(row, "BIRTHDATE", "DOB", "DATE_OF_BIRTH", "BIRTH_DATE")
    gender     = get_value(row, "GENDER", "SEX")

    # Format the date nicely
    try:
        dob = pd.to_datetime(dob).strftime("%d-%m-%Y")
    except:
        pass

    return {
        "patient_id": patient_id,
        "full_name":  first_name + " " + last_name,
        "dob":        dob,
        "gender":     gender.capitalize(),
    }

#  OUTPUT 2 — ACTIVE CONDITIONS COUNT
#  Count conditions where the STOP date is empty (still active)

def get_active_conditions(tables, patient_id):
    conditions_table = tables["conditions"]
    id_col           = find_id_column(conditions_table)

    # Get only this patient's conditions
    patient_conditions = conditions_table[
        conditions_table[id_col].astype(str).str.strip() == patient_id
    ]

    # Find the column that says when a condition ended
    stop_col = find_column(patient_conditions, "STOP", "END", "END_DATE", "STOP_DATE")

    # If STOP is empty (NaN) the condition is still active
    if stop_col:
        active_conditions = patient_conditions[patient_conditions[stop_col].isna()]
    else:
        active_conditions = patient_conditions

    # Get the list of condition names
    desc_col = find_column(active_conditions, "DESCRIPTION", "DESC", "CONDITION")
    condition_names = active_conditions[desc_col].dropna().tolist() if desc_col else []

    return {
        "active_conditions_count": len(active_conditions),
        "active_conditions_list":  condition_names,
    }

#  OUTPUT 3 — ENCOUNTER COUNT
#  Count how many times this patient visited the hospital

def get_encounter_count(tables, patient_id):
    encounters_table = tables["encounters"]
    id_col           = find_id_column(encounters_table)

    patient_encounters = encounters_table[
        encounters_table[id_col].astype(str).str.strip() == patient_id
    ]

    return {"encounter_count": len(patient_encounters)}

#  OUTPUT 4 — RISK SCORE
#  Calculate risk based on number of encounters
#    1-3  encounters = score 2  (low risk)
#    4-9  encounters = score 5  (medium risk)
#    10+  encounters = score 9  (high risk)

def calculate_risk_score(encounter_count):
    if encounter_count == 0:
        risk_score = 0
    elif encounter_count <= 3:
        risk_score = 2
    elif encounter_count <= 9:
        risk_score = 5
    else:
        risk_score = 9

    return {"risk_score": risk_score}

#  OUTPUT 5 — NEW RISK SCORE
#  Combined score = risk score + number of active conditions

def calculate_new_risk_score(risk_score, active_conditions_count):
    return {"new_risk_score": risk_score + active_conditions_count}

#  OUTPUT 6 — OPENFDA DRUG LOOKUP
# 3. API Call - look up drug info on OpenFDA API ##################################
#  Look up each medication on the OpenFDA API
#  Returns: purpose, warnings, side effects
#  No API key needed — completely free!

def clean_drug_name(full_drug_name):
    # Strip dosage info so OpenFDA can find the drug
    # Example: "Hydrochlorothiazide 25 MG Oral Tablet" → "Hydrochlorothiazide"
    cleaned = re.split(
        r'\s+\d+[\d./]*\s*(MG|ML|MCG|IU|%|Day|tablet|capsule|oral|pack|system)',
        full_drug_name, flags=re.IGNORECASE
    )[0].strip()

    cleaned = re.sub(
        r'\s+(oral|tablet|capsule|solution|suspension|injection|pack|system|cream|patch).*$',
        '', cleaned, flags=re.IGNORECASE
    ).strip()

    return cleaned if cleaned else full_drug_name


def lookup_drug_on_fda(drug_name):
    # Try searching by brand name first, then by generic name
    cleaned_name = clean_drug_name(drug_name)
    print("      (searching FDA for: " + cleaned_name + ")")

    search_options = [
        ("openfda.brand_name",   cleaned_name),
        ("openfda.generic_name", cleaned_name),
        ("openfda.brand_name",   cleaned_name.split()[0]),
        ("openfda.generic_name", cleaned_name.split()[0]),
    ]

    for search_field, search_term in search_options:
        try:
            response = requests.get(
                OPENFDA_URL,
                params={"search": search_field + ':"' + search_term + '"', "limit": 1},
                timeout=10
            )

            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    drug_label = results[0]

                    def get_first(key):
                        value = drug_label.get(key, ["N/A"])
                        return value[0][:400]

                    return {
                        "drug":              drug_name,
                        "purpose":           get_first("purpose"),
                        "warnings":          get_first("warnings"),
                        "adverse_reactions": get_first("adverse_reactions"),
                    }
        except requests.RequestException:
            pass

    return {"drug": drug_name, "error": "Drug not found in FDA database"}


def get_medications_with_fda_info(tables, patient_id):
    medications_table = tables["medications"]
    id_col            = find_id_column(medications_table)

    patient_meds = medications_table[
        medications_table[id_col].astype(str).str.strip() == patient_id
    ]

    if patient_meds.empty:
        return {"medications_fda": []}

    name_col   = find_column(patient_meds, "DESCRIPTION", "DRUG", "MEDICATION", "NAME")
    stop_col   = find_column(patient_meds, "STOP", "STOPDATE", "END_DATE")
    reason_col = find_column(patient_meds, "REASONDESCRIPTION", "REASON", "INDICATION")

    if not name_col:
        return {"medications_fda": []}

    # Only show currently active medications (where STOP is empty)
    if stop_col:
        active_meds = patient_meds[patient_meds[stop_col].isna()]
    else:
        active_meds = patient_meds

    # If no active meds found, fall back to most recent 5
    if active_meds.empty:
        active_meds = patient_meds.head(5)

    # Remove duplicate drug names
    unique_meds = active_meds.drop_duplicates(subset=[name_col])

    results = []
    for index, med_row in unique_meds.head(8).iterrows():
        drug_name  = str(med_row[name_col])
        csv_reason = str(med_row[reason_col]) if reason_col and pd.notna(med_row.get(reason_col)) else ""

        print("    Looking up: " + drug_name[:60])
        fda_info = lookup_drug_on_fda(drug_name)

        # Use the reason from our CSV if available — more specific than FDA
        if csv_reason and csv_reason.lower() not in ("nan", "n/a", ""):
            fda_info["purpose"] = csv_reason

        results.append(fda_info)

    return {"medications_fda": results}

#  OUTPUT 7 — SYMPTOM CHECKER USING GPT-4o-mini
# Lab/vital sign keywords to EXCLUDE from symptoms

EXCLUDE_FROM_SYMPTOMS = [
    "height", "weight", "bmi", "body mass", "blood pressure", "diastolic",
    "systolic", "heart rate", "respiratory rate", "oxygen saturation",
    "leukocytes", "hemoglobin", "hematocrit", "platelets", "cholesterol",
    "creatinine", "glucose", "triglycerides", "score]", "per age and sex",
]

# Survey fields that ARE relevant to include
USEFUL_SURVEY_FIELDS = [
    "tobacco", "stress", "afraid of your partner", "pain severity", "phq", "gad",
]

# Symptom keywords to scan for in conditions
SYMPTOM_KEYWORDS = [
    "fever", "cough", "headache", "fatigue", "nausea", "vomiting", "diarrhea",
    "pain", "shortness of breath", "dizziness", "rash", "sore throat", "chills",
    "swelling", "anxiety", "depression", "insomnia", "hypertension", "diabetes",
    "obesity", "smoking", "abuse", "infection",
]


def extract_symptoms_from_data(tables, patient_id):
    symptoms = []

    # --- Get symptoms from active CONDITIONS ---
    conditions_table = tables["conditions"]
    id_col           = find_id_column(conditions_table)
    patient_conditions = conditions_table[
        conditions_table[id_col].astype(str).str.strip() == patient_id
    ]

    stop_col = find_column(patient_conditions, "STOP", "END", "END_DATE")
    active   = patient_conditions[patient_conditions[stop_col].isna()] if stop_col else patient_conditions
    desc_col = find_column(active, "DESCRIPTION", "DESC", "CONDITION")

    if desc_col:
        for desc in active[desc_col].dropna().tolist():
            # Skip pure social/admin entries
            if not any(skip in desc.lower() for skip in ["education", "unemployed", "finding"]):
                symptoms.append(desc)

    # --- Get symptoms from OBSERVATIONS (surveys and scores) ---
    observations_table = tables["observations"]
    obs_id_col         = find_id_column(observations_table)
    patient_obs = observations_table[
        observations_table[obs_id_col].astype(str).str.strip() == patient_id
    ]

    obs_desc_col  = find_column(patient_obs, "DESCRIPTION", "DESC")
    obs_value_col = find_column(patient_obs, "VALUE")

    if obs_desc_col:
        for index, row in patient_obs.iterrows():
            desc  = str(row[obs_desc_col]).lower() if pd.notna(row[obs_desc_col]) else ""
            value = str(row[obs_value_col]).strip() if obs_value_col and pd.notna(row.get(obs_value_col)) else ""

            # Skip pure measurements
            if any(exclude in desc for exclude in EXCLUDE_FROM_SYMPTOMS):
                continue

            # Include useful survey results
            if any(useful in desc for useful in USEFUL_SURVEY_FIELDS):
                if "pain severity" in desc:
                    try:
                        if float(value) > 0:
                            symptoms.append("Pain severity: " + value + "/10")
                    except ValueError:
                        pass
                elif "phq" in desc:
                    try:
                        if float(value) > 0:
                            symptoms.append("depression")
                    except ValueError:
                        pass
                elif "gad" in desc:
                    try:
                        if float(value) > 0:
                            symptoms.append("anxiety")
                    except ValueError:
                        pass
                elif "tobacco" in desc and value.lower() not in ("never smoker", "0", ""):
                    symptoms.append("smoking")

    # --- Keyword scan for extra coverage ---
    all_text = " ".join(symptoms).lower()
    for keyword in SYMPTOM_KEYWORDS:
        if keyword in all_text and keyword not in " ".join(symptoms).lower():
            symptoms.append(keyword.capitalize())

    # Remove duplicates while keeping the order
    seen       = set()
    no_dupes   = []
    for symptom in symptoms:
        if symptom.lower() not in seen:
            seen.add(symptom.lower())
            no_dupes.append(symptom)

    return no_dupes


def analyse_symptoms_with_gpt(openai_client, symptoms, patient_details):
    if not symptoms:
        return {
            "possible_causes": [],
            "severity":        "No symptoms found",
            "triage_guidance": "No action needed",
            "source":          "none",
        }

    age    = calculate_age(patient_details.get("dob", ""))
    gender = patient_details.get("gender", "Unknown")

    # Ask GPT to analyse the symptoms and return structured JSON
    prompt = (
        "You are a clinical triage assistant. Analyse the symptoms below "
        "and return ONLY a JSON object — no extra text, no markdown.\n\n"
        "Patient: " + str(age) + " year old " + gender + "\n"
        "Symptoms: " + ", ".join(symptoms) + "\n\n"
        "Return exactly this JSON structure:\n"
        "{\n"
        '  "possible_causes": [\n'
        '    {"disease": "Disease Name", "likelihood": "35%", "emergency": false}\n'
        "  ],\n"
        '  "severity": "LOW or MODERATE or HIGH — one sentence explanation",\n'
        '  "triage_guidance": "Specific actionable next steps for this patient"\n'
        "}\n\n"
        "Rules:\n"
        "- List the top 4-5 most likely conditions\n"
        "- Mark emergency as true only for life-threatening conditions\n"
        "- Severity must start with LOW, MODERATE, or HIGH\n"
        "- Return valid JSON only"
    )

    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a clinical triage AI. Return only valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,   # low = more consistent output
            max_tokens=400,
        )

        response_text = response.choices[0].message.content.strip()

        # Remove markdown code fences if GPT added them
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        result = json.loads(response_text)
        result["source"] = "gpt-4o-mini"
        return result

    except json.JSONDecodeError:
        return {
            "possible_causes": [],
            "severity":        "Could not parse response",
            "triage_guidance": "Please consult a general practitioner",
            "source":          "parse_error",
        }
    except Exception as error:
        return {
            "possible_causes": [],
            "severity":        "Unable to determine",
            "triage_guidance": "Please consult a general practitioner",
            "source":          "error: " + str(error)[:60],
        }

#  4. API CallExtract symptoms from the data and ask GPT to analyse them
def get_symptom_intelligence(tables, patient_id, patient_details, openai_client):
    # Step 1: Extract raw symptoms from CSV data
    raw_symptoms = extract_symptoms_from_data(tables, patient_id)
    print("    Raw symptoms found: " + str(raw_symptoms))

    # Step 2: Send to GPT for analysis
    print("    Sending to GPT-4o-mini for analysis...")
    intelligence = analyse_symptoms_with_gpt(openai_client, raw_symptoms, patient_details)

    return {
        "symptoms_found":       raw_symptoms,
        "symptom_intelligence": intelligence,
    }

#  OUTPUT 8 — CLINICAL SUMMARY USING GPT-4o-mini
# 5. API Call - Generate a clinical summary using GPT-4o-mini ##################################
#  Combine all data and generate a human-readable summary

def generate_clinical_summary(openai_client, rag_context, all_data):
    patient    = all_data["patient_details"]
    conditions = all_data["active_conditions"]
    encounters = all_data["encounter_info"]
    risk       = all_data["risk_scores"]
    meds       = all_data["medications_fda"].get("medications_fda", [])
    symptoms   = all_data["symptom_checker"].get("symptoms_found", [])
    intel      = all_data["symptom_checker"].get("symptom_intelligence", {})

    age = calculate_age(patient.get("dob", ""))

    # Format medications as readable text
    med_lines = ""
    for med in meds:
        med_lines += "  - " + med.get("drug", "Unknown") + ": " + med.get("purpose", "N/A")[:100] + "\n"
    if not med_lines:
        med_lines = "  None on record"

    # Format conditions as readable text
    condition_lines = ""
    for cond in conditions.get("active_conditions_list", [])[:15]:
        condition_lines += "  - " + cond + "\n"
    if not condition_lines:
        condition_lines = "  None listed"

    # Format possible causes from symptom intelligence
    causes_text = ""
    for cause in intel.get("possible_causes", [])[:5]:
        causes_text += "  - " + cause.get("disease", "") + " (" + cause.get("likelihood", "") + ")\n"
    if not causes_text:
        causes_text = "  Not available"

    # Build the full prompt for GPT
    prompt = (
        "Generate a professional 4-6 sentence clinical summary using ONLY the data below.\n"
        "Do not invent any details. Use clear medical language.\n\n"
        "Patient    : " + patient["full_name"] + " | ID: " + patient["patient_id"] + "\n"
        "Age/DOB    : " + patient["dob"] + " (Age: " + str(age) + " years) | Gender: " + patient["gender"] + "\n"
        "Conditions : " + str(conditions["active_conditions_count"]) + " active\n"
        + condition_lines +
        "Encounters : " + str(encounters["encounter_count"]) + "\n"
        "Risk Score : " + str(risk["risk_score"]) + " | New Risk Score: " + str(risk["new_risk_score"]) + "\n"
        "Medications:\n" + med_lines +
        "Symptoms   : " + (", ".join(symptoms) if symptoms else "None identified") + "\n\n"
        "Possible Causes (from symptom analysis):\n" + causes_text +
        "Severity       : " + intel.get("severity", "N/A") + "\n"
        "Triage Guidance: " + intel.get("triage_guidance", "N/A") + "\n\n"
        "Retrieved Patient Records:\n" + rag_context[:2500] + "\n\n"
        "Write the clinical summary now (no headers, no bullet points):"
    )

    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a clinical documentation AI. Write accurate, concise clinical summaries."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()

#  MAIN AGENT FUNCTION
#  Runs all 8 outputs and returns the final JSON

def run_agent(patient_id):
    print("\n============================================")
    print("  CLINICAL SUMMARY AGENT")
    print("  Patient ID: " + patient_id)
    print("  Model: " + GPT_MODEL)
    print("============================================\n")

    # Check the API key is set
    if not MY_API_KEY:
        raise EnvironmentError(
            "OpenAI API key is not set!\n"
            "Windows:   $env:OPENAI_API_KEY='sk-...'\n"
            "Mac/Linux: export OPENAI_API_KEY='sk-...'"
        )

    # Set up our clients
    openai_client = OpenAI(api_key=MY_API_KEY)

    print("[Loading] Reading CSV files and connecting to ChromaDB...")
    tables     = load_csv_files()
    collection = connect_to_chromadb()

    # Output 1: Patient details
    print("\n[1] Getting patient details...")
    patient_details = get_patient_details(tables, patient_id)
    print("    " + patient_details["full_name"] + " | " + patient_details["dob"] + " | " + patient_details["gender"])

    # Output 2: Active conditions
    print("\n[2] Counting active conditions...")
    active_conditions = get_active_conditions(tables, patient_id)
    print("    " + str(active_conditions["active_conditions_count"]) + " active conditions")

    # Output 3: Encounter count
    print("\n[3] Counting encounters...")
    encounter_info = get_encounter_count(tables, patient_id)
    print("    " + str(encounter_info["encounter_count"]) + " total encounters")

    # Output 4: Risk score
    print("\n[4] Calculating risk score...")
    risk_score = calculate_risk_score(encounter_info["encounter_count"])
    print("    Risk score: " + str(risk_score["risk_score"]))

    # Output 5: New risk score
    print("\n[5] Calculating new risk score...")
    new_risk = calculate_new_risk_score(
        risk_score["risk_score"],
        active_conditions["active_conditions_count"]
    )
    print("    New risk score: " + str(new_risk["new_risk_score"]))

    # Output 6: OpenFDA drug info
    print("\n[6] Looking up medications on OpenFDA...")
    medications_info = get_medications_with_fda_info(tables, patient_id)
    print("    " + str(len(medications_info.get("medications_fda", []))) + " drugs looked up")

    # Output 7: Symptom checker
    print("\n[7] Analysing symptoms with GPT-4o-mini...")
    symptom_data = get_symptom_intelligence(tables, patient_id, patient_details, openai_client)
    intel        = symptom_data.get("symptom_intelligence", {})
    print("    Severity : " + intel.get("severity", "N/A"))
    print("    Triage   : " + intel.get("triage_guidance", "N/A"))

    # RAG: Retrieve relevant context from ChromaDB
    print("\n[RAG] Retrieving relevant records from ChromaDB...")
    rag_context = retrieve_patient_context(openai_client, collection, patient_id)
    print("    Retrieved " + str(len(rag_context)) + " characters of context")

    # Output 8: GPT clinical summary
    print("\n[8] Generating clinical summary with GPT-4o-mini...")
    all_data = {
        "patient_details":   patient_details,
        "active_conditions": active_conditions,
        "encounter_info":    encounter_info,
        "risk_scores": {
            "risk_score":     risk_score["risk_score"],
            "new_risk_score": new_risk["new_risk_score"],
        },
        "medications_fda": medications_info,
        "symptom_checker": symptom_data,
    }
    summary = generate_clinical_summary(openai_client, rag_context, all_data)
    print("    Summary generated (" + str(len(summary)) + " characters)")

    # Build the final output object
    final_output = {
        "patient_id":              patient_details["patient_id"],
        "full_name":               patient_details["full_name"],
        "dob":                     patient_details["dob"],
        "gender":                  patient_details["gender"],
        "active_conditions_count": active_conditions["active_conditions_count"],
        "active_conditions_list":  active_conditions["active_conditions_list"],
        "encounter_count":         encounter_info["encounter_count"],
        "risk_score":              risk_score["risk_score"],
        "new_risk_score":          new_risk["new_risk_score"],
        "medications": [
            {
                "drug":              med.get("drug"),
                "purpose":           med.get("purpose",           "N/A"),
                "warnings":          med.get("warnings",          "N/A"),
                "adverse_reactions": med.get("adverse_reactions", "N/A"),
            }
            for med in medications_info.get("medications_fda", [])
        ],
        "symptoms_found":       symptom_data.get("symptoms_found", []),
        "symptom_intelligence": symptom_data.get("symptom_intelligence", {}),
        "clinical_summary":     summary,
    }

    return final_output

#  ENTRY POINT
#  This runs when you do: python clinical_summary_agent.py ID

if __name__ == "__main__":
    # Check they passed a patient ID when running the script
    if len(sys.argv) < 2:
        print("\nUsage  : python clinical_summary_agent.py PATIENT_ID")
        print("Example: python clinical_summary_agent.py b9c610cd-28a6-4636-ccb6-c7a0d2a4cb85\n")
        sys.exit(1)

    patient_id = sys.argv[1]

    try:
        result = run_agent(patient_id)

        # Print the final output
        print("\n============================================")
        print("  FINAL OUTPUT")
        print("============================================")
        print(json.dumps(result, indent=2))

        # Also save it to a JSON file
        output_file = "summary_" + patient_id[:8] + ".json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print("\n  Saved to: " + output_file + "\n")

    except (FileNotFoundError, KeyError, ValueError) as error:
        print("\nERROR: " + str(error))
        sys.exit(1)
