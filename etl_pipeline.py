#  ETL PIPELINE
#  What this file does:
#    1. Reads 5 CSV files from the data/ folder
#    2. Converts each patient's data into small text chunks
#    3. Sends those chunks to OpenAI to get vectors (numbers)
#    4. Saves those vectors into ChromaDB (a local database)
#
#  You only need to run this ONCE before running the agent.
#  Run it like this:   python etl_pipeline.py

import os
import json
import time
import hashlib
import re
from datetime import datetime
from collections import Counter

import pandas as pd
from tqdm import tqdm
import chromadb
from openai import OpenAI

CSV_FOLDER       = "./data"           # 5 CSV files 
CHROMA_FOLDER    = "./chroma_db"      # ChromaDB will save data
COLLECTION_NAME  = "clinical_records" # name of our  new table in ChromaDB
OPENAI_EMB_MODEL = "text-embedding-3-small" # model that converts text to numbers
BATCH_SIZE       = 20                 # how many chunks to send to OpenAI at once
MAX_CHARS        = 1500               # max characters in one chunk

MY_API_KEY = os.environ.get("OPENAI_API_KEY", "")

#  STEP 1 — READ ALL CSV FILES

def read_all_csv_files(csv_folder):
    files_we_need = {
        "patients":     "ID",
        "encounters":   "PATIENT",
        "conditions":   "PATIENT",
        "medications":  "PATIENT",
        "observations": "PATIENT",
    }

    loaded_tables = {}  # we will store all loaded tables here

    print("\n STEP 1: Reading CSV files")

    for file_name, required_column in files_we_need.items():
        # Build the full path like "./data/patients.csv"
        full_path = os.path.join(csv_folder, file_name + ".csv")

        # Check the file actually exists before trying to open it
        if not os.path.exists(full_path):
            print("ERROR: Could not find " + full_path)
            print("Please make sure all 5 CSV files are in the data/ folder")
            raise FileNotFoundError("Missing file: " + full_path)

        # Read the CSV file into a pandas table (like an Excel sheet)
        table = pd.read_csv(full_path, low_memory=False)

        # Make all column names UPPERCASE so "Patient" and "PATIENT" both work
        table.columns = [col.strip().upper() for col in table.columns]

        # Check the column we need actually exists in this file
        column_found = any(required_column in col for col in table.columns)
        if not column_found:
            print("ERROR: " + file_name + ".csv does not have a " + required_column + " column")
            raise ValueError("Missing column in " + file_name + ".csv")

        loaded_tables[file_name] = table
        print("  Loaded " + file_name + ".csv — " + str(len(table)) + " rows")

    return loaded_tables

#  Small reusable tools used in Step 2

def find_patient_id_column(table):
    # Different CSV files use different names for the patient ID column
    # This function finds whichever one exists
    for col in table.columns:
        if col in ("ID", "PATIENT", "PATIENT_ID"):
            return col
    for col in table.columns:
        if "PATIENT" in col or col == "ID":
            return col
    raise KeyError("Cannot find patient ID column in table: " + str(table.columns.tolist()))


def find_column(table, *possible_names):
    # Try each possible column name and return the first one that exists
    # Example: find_column(table, "STOP", "END", "END_DATE")
    for name in possible_names:
        if name in table.columns:
            return name
    return None  # none of those columns exist


def get_cell_value(row, *possible_columns):
    # Try to get a value from a row using different possible column names
    # Returns "N/A" if none of them exist or the value is empty
    for col in possible_columns:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return "N/A"


def make_unique_id(patient_id, section_name, chunk_number=0):
    # Create a unique ID for each chunk using MD5 hashing
    # Same input always gives same ID: safe to re-run ETL without duplicates
    text_to_hash = patient_id + "::" + section_name + "::" + str(chunk_number)
    return hashlib.md5(text_to_hash.encode()).hexdigest()


def split_long_text(text, max_chars=MAX_CHARS):
    # If text is too long, split into smaller overlapping pieces
    # We overlap by 200 chars so no context is lost at the edges
    if len(text) <= max_chars:
        return [text]  # short enough, no splitting needed

    pieces = []
    start = 0
    while start < len(text):
        end = start + max_chars
        pieces.append(text[start:end])
        start = start + max_chars - 200  # go back 200 chars for overlap
    return pieces

#  STEP 2 — BUILD TEXT CHUNKS FOR EACH PATIENT
#  Convert each patient's table rows into readable text blocks

def build_demographics_text(patient_id, patients_table):
    id_col = find_patient_id_column(patients_table)
    matching_rows = patients_table[patients_table[id_col].astype(str).str.strip() == patient_id]

    if matching_rows.empty:
        return []

    row = matching_rows.iloc[0]  # get the first (and usually only) matching row
    first_name = get_cell_value(row, "FIRST", "FIRSTNAME")
    last_name  = get_cell_value(row, "LAST",  "LASTNAME")
    dob        = get_cell_value(row, "BIRTHDATE", "DOB", "DATE_OF_BIRTH")
    gender     = get_cell_value(row, "GENDER", "SEX")
    city       = get_cell_value(row, "CITY")
    state      = get_cell_value(row, "STATE")

    text = (
        "PATIENT DEMOGRAPHICS\n"
        "Patient ID : " + patient_id + "\n"
        "Full Name  : " + first_name + " " + last_name + "\n"
        "DOB        : " + dob + "\n"
        "Gender     : " + gender + "\n"
        "Location   : " + city + ", " + state
    )

    return [{
        "id":         make_unique_id(patient_id, "demographics"),
        "text":       text,
        "section":    "demographics",
        "patient_id": patient_id,
    }]


def build_conditions_text(patient_id, conditions_table):
    id_col = find_patient_id_column(conditions_table)
    patient_rows = conditions_table[conditions_table[id_col].astype(str).str.strip() == patient_id]

    if patient_rows.empty:
        return []

    stop_col  = find_column(patient_rows, "STOP", "END", "END_DATE")
    desc_col  = find_column(patient_rows, "DESCRIPTION", "DESC", "CONDITION")
    start_col = find_column(patient_rows, "START", "START_DATE", "ONSET")

    lines = []
    for index, row in patient_rows.iterrows():
        description = str(row[desc_col])       if desc_col  else "Unknown"
        start_date  = str(row[start_col])[:10] if start_col else "N/A"

        # If STOP date is empty (NaN) the condition is still ACTIVE
        if stop_col and pd.notna(row.get(stop_col)):
            status = "Resolved " + str(row[stop_col])
        else:
            status = "ACTIVE"

        lines.append("  - " + description + " | Since: " + start_date + " | Status: " + status)

    full_text = "CONDITIONS for Patient " + patient_id + ":\n" + "\n".join(lines)

    chunks = []
    for i, piece in enumerate(split_long_text(full_text)):
        chunks.append({
            "id":         make_unique_id(patient_id, "conditions", i),
            "text":       piece,
            "section":    "conditions",
            "patient_id": patient_id,
        })
    return chunks


def build_encounters_text(patient_id, encounters_table):
    id_col = find_patient_id_column(encounters_table)
    patient_rows = encounters_table[encounters_table[id_col].astype(str).str.strip() == patient_id]

    if patient_rows.empty:
        return []

    date_col   = find_column(patient_rows, "START", "DATE", "ENCOUNTERDATE")
    type_col   = find_column(patient_rows, "ENCOUNTERCLASS", "CLASS", "TYPE")
    reason_col = find_column(patient_rows, "REASONDESCRIPTION", "REASON")

    lines = []
    for index, row in patient_rows.iterrows():
        date       = str(row[date_col])[:10] if date_col else "N/A"
        visit_type = str(row[type_col])      if type_col else "N/A"
        reason     = str(row[reason_col]) if reason_col and pd.notna(row.get(reason_col)) else "N/A"
        lines.append("  - " + date + " | Type: " + visit_type + " | Reason: " + reason)

    total = str(len(patient_rows))
    full_text = "ENCOUNTERS for Patient " + patient_id + " (Total: " + total + "):\n" + "\n".join(lines)

    chunks = []
    for i, piece in enumerate(split_long_text(full_text)):
        chunks.append({
            "id":         make_unique_id(patient_id, "encounters", i),
            "text":       piece,
            "section":    "encounters",
            "patient_id": patient_id,
        })
    return chunks


def build_medications_text(patient_id, medications_table):
    id_col = find_patient_id_column(medications_table)
    patient_rows = medications_table[medications_table[id_col].astype(str).str.strip() == patient_id]

    if patient_rows.empty:
        return []

    name_col   = find_column(patient_rows, "DESCRIPTION", "DRUG", "MEDICATION", "NAME")
    start_col  = find_column(patient_rows, "START", "STARTDATE")
    stop_col   = find_column(patient_rows, "STOP",  "STOPDATE", "END_DATE")
    reason_col = find_column(patient_rows, "REASONDESCRIPTION", "REASON")

    lines = []
    for index, row in patient_rows.iterrows():
        drug      = str(row[name_col])        if name_col  else "Unknown"
        start     = str(row[start_col])[:10]  if start_col else "N/A"
        stop      = str(row[stop_col])[:10]   if stop_col and pd.notna(row.get(stop_col)) else "ONGOING"
        reason    = str(row[reason_col]) if reason_col and pd.notna(row.get(reason_col)) else "N/A"
        lines.append("  - " + drug + " | " + start + " to " + stop + " | For: " + reason)

    full_text = "MEDICATIONS for Patient " + patient_id + ":\n" + "\n".join(lines)

    chunks = []
    for i, piece in enumerate(split_long_text(full_text)):
        chunks.append({
            "id":         make_unique_id(patient_id, "medications", i),
            "text":       piece,
            "section":    "medications",
            "patient_id": patient_id,
        })
    return chunks


def build_observations_text(patient_id, observations_table):
    id_col = find_patient_id_column(observations_table)
    patient_rows = observations_table[observations_table[id_col].astype(str).str.strip() == patient_id]

    if patient_rows.empty:
        return []

    date_col  = find_column(patient_rows, "DATE")
    desc_col  = find_column(patient_rows, "DESCRIPTION", "DESC")
    value_col = find_column(patient_rows, "VALUE")
    unit_col  = find_column(patient_rows, "UNITS", "UNIT")
    cat_col   = find_column(patient_rows, "CATEGORY")

    lines = []
    for index, row in patient_rows.iterrows():
        date     = str(row[date_col])[:10] if date_col  else "N/A"
        desc     = str(row[desc_col])      if desc_col  else "N/A"
        value    = str(row[value_col])     if value_col else ""
        unit     = str(row[unit_col])      if unit_col  else ""
        category = str(row[cat_col])       if cat_col   else ""
        lines.append("  - [" + date + "] " + desc + ": " + value + " " + unit + " (Category: " + category + ")")

    full_text = "OBSERVATIONS for Patient " + patient_id + ":\n" + "\n".join(lines)

    chunks = []
    for i, piece in enumerate(split_long_text(full_text)):
        chunks.append({
            "id":         make_unique_id(patient_id, "observations", i),
            "text":       piece,
            "section":    "observations",
            "patient_id": patient_id,
        })
    return chunks


def build_all_chunks(all_tables):
    print("\n STEP 2: Building text chunks for each patient")

    patients_table  = all_tables["patients"]
    id_col          = find_patient_id_column(patients_table)
    all_patient_ids = patients_table[id_col].astype(str).str.strip().unique().tolist()

    print("  Found " + str(len(all_patient_ids)) + " patients to process")

    all_chunks = []

    for patient_id in tqdm(all_patient_ids, desc="  Building chunks"):
        all_chunks += build_demographics_text(patient_id, all_tables["patients"])
        all_chunks += build_conditions_text(patient_id,   all_tables["conditions"])
        all_chunks += build_encounters_text(patient_id,   all_tables["encounters"])
        all_chunks += build_medications_text(patient_id,  all_tables["medications"])
        all_chunks += build_observations_text(patient_id, all_tables["observations"])

    # Show how many chunks were created per section
    section_counts = Counter(chunk["section"] for chunk in all_chunks)
    print("\n  Total chunks created: " + str(len(all_chunks)))
    for section, count in section_counts.items():
        print("    " + section + ": " + str(count) + " chunks")

    return all_chunks

#  STEP 3 — EMBED AND SAVE TO CHROMADB
#  Convert text chunks into numbers and save to ChromaDB

def setup_chromadb(chroma_folder):
    # Create the folder if it doesn't exist yet
    os.makedirs(chroma_folder, exist_ok=True)

    # Connect to ChromaDB, PersistentClient saves to disk
    chroma_client = chromadb.PersistentClient(path=chroma_folder)

    # Create or open our collection (like a table in a database)
    # hnsw:space = cosine means we measure similarity with cosine distance
    # which works best for comparing text embeddings
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("  ChromaDB is ready at: " + chroma_folder)
    return collection

# 1. API call to OpenAI to convert text to vectors  ##################################

def convert_text_to_vectors(openai_client, list_of_texts, max_retries=5):
    # Send texts to OpenAI and get back vectors (lists of numbers)
    # Has retry logic in case we hit the rate limit
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                model=OPENAI_EMB_MODEL,
                input=list_of_texts
            )
            # Extract just the numbers from each response item
            return [item.embedding for item in response.data]

        except Exception as error:
            error_text = str(error)

            # Check if this is a rate limit error (too many requests)
            if "rate_limit" in error_text.lower() or "429" in error_text:
                # Wait longer each time we retry (exponential backoff)
                wait_time = 2 ** attempt + 1   # 2s, 3s, 5s, 9s, 17s

                # OpenAI sometimes tells us exactly how long to wait
                ms_match = re.search(r"try again in (\d+)ms", error_text)
                if ms_match:
                    wait_time = int(ms_match.group(1)) / 1000 + 0.5

                print("  Rate limit hit — waiting " + str(round(wait_time, 1)) + "s before retry " + str(attempt + 1) + "/" + str(max_retries))
                time.sleep(wait_time)
            else:
                raise  # some other error — stop immediately

    raise RuntimeError("Failed to get embeddings after " + str(max_retries) + " retries")


def save_chunks_to_chromadb(all_chunks, collection, openai_client):
    print("\n STEP 3: Converting text to vectors and saving to ChromaDB")
    print("  Total chunks to process: " + str(len(all_chunks)))
    print("  Sending " + str(BATCH_SIZE) + " chunks at a time to OpenAI")

    # Process in small batches to avoid hitting the rate limit
    for batch_start in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="  Embedding and saving"):

        # Get the current batch of chunks
        batch = all_chunks[batch_start : batch_start + BATCH_SIZE]

        # Separate out the pieces ChromaDB needs
        texts     = [chunk["text"]       for chunk in batch]
        ids       = [chunk["id"]         for chunk in batch]
        metadata  = [
            {
                "patient_id": chunk["patient_id"],
                "section":    chunk["section"]
            }
            for chunk in batch
        ]

        # Convert texts to vectors using OpenAI
        vectors = convert_text_to_vectors(openai_client, texts)

        # Save to ChromaDB
        # upsert = insert if new, update if ID already exists
        # This makes it safe to re-run the ETL without duplicates
        collection.upsert(
            ids        = ids,
            embeddings = vectors,
            documents  = texts,
            metadatas  = metadata
        )

        # Small pause between batches to be polite to the API
        time.sleep(0.3)

    print("  Total chunks saved in ChromaDB: " + str(collection.count()))

#  MAIN FUNCTION
#  Runs all 3 steps in order

def run_etl_pipeline():
    start_time = datetime.now()

    print("\n============================================")
    print("  CLINICAL ETL PIPELINE — Starting")
    print("============================================")

    # Check the API key is set before doing anything
    if not MY_API_KEY:
        print("\nERROR: OpenAI API key is not set!")
        print("  Windows:   $env:OPENAI_API_KEY='sk-...'")
        raise EnvironmentError("OPENAI_API_KEY not set")

    # Create the OpenAI client we will use for embeddings
    openai_client = OpenAI(api_key=MY_API_KEY)

    # Step 1: Read all CSV files
    all_tables = read_all_csv_files(CSV_FOLDER)

    # Step 2: Build text chunks for every patient
    all_chunks = build_all_chunks(all_tables)

    # Step 3: Embed and save to ChromaDB
    collection = setup_chromadb(CHROMA_FOLDER)
    save_chunks_to_chromadb(all_chunks, collection, openai_client)

    # Save a simple audit log of what was stored
    audit_log = []
    for chunk in all_chunks:
        audit_log.append({
            "id":         chunk["id"],
            "patient_id": chunk["patient_id"],
            "section":    chunk["section"]
        })
    audit_path = os.path.join(CHROMA_FOLDER, "audit_log.json")
    with open(audit_path, "w") as f:
        json.dump(audit_log, f, indent=2)
    print("\n  Audit log saved to: " + audit_path)

    # Show how long the whole pipeline took
    elapsed = (datetime.now() - start_time).seconds
    print("\n============================================")
    print("  ETL COMPLETE! Took " + str(elapsed) + " seconds")
    print("  Next step: python clinical_summary_agent.py YOUR_PATIENT_ID")
    print("============================================\n")


# Run: python etl_pipeline.py
if __name__ == "__main__":
    run_etl_pipeline()
