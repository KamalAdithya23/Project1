import os
import sqlite3
import json
import re
import pytesseract
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException
from PIL import Image
import numpy as np
# Ensure all file operations are within /data/
DATA_DIR = "data/"

app = FastAPI()

def enforce_security(file_path):
    if not file_path.startswith(DATA_DIR):
        raise PermissionError(f"Access denied: {file_path} is outside {DATA_DIR}")

# A3: Count Wednesdays in /data/dates.txt
date_formats = [
    "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%m-%d-%Y %I:%M %p"
]

def parse_date(date_str):
    date_str = date_str.strip()
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def count_wednesdays():
    input_file = os.path.join(DATA_DIR, "dates.txt")
    output_file = os.path.join(DATA_DIR, "dates-wednesdays.txt")
    enforce_security(input_file)
    enforce_security(output_file)
    
    if not os.path.exists(input_file):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("0\n")
        return
    
    with open(input_file, "r", encoding="utf-8") as f:
        dates = f.readlines()
    
    wednesday_count = sum(
        1 for date in dates if (parsed_date := parse_date(date)) and parsed_date.weekday() == 2
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(wednesday_count) + "\n")

# A4: Sort contacts by last and first name
def sort_contacts():
    input_file = os.path.join(DATA_DIR, "contacts.json")
    output_file = os.path.join(DATA_DIR, "contacts-sorted.json")
    enforce_security(input_file)
    enforce_security(output_file)

    with open(input_file, "r", encoding="utf-8") as f:
        contacts = json.load(f)

    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_contacts, f, indent=4)

# A5: Get first lines of the 10 most recent log files
def recent_logs():
    logs_dir = os.path.join(DATA_DIR, "logs")
    output_file = os.path.join(DATA_DIR, "logs-recent.txt")
    enforce_security(output_file)

    log_files = sorted([os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")], key=os.path.getmtime, reverse=True)[:10]

    first_lines = []
    for log in log_files:
        enforce_security(log)
        with open(log, "r", encoding="utf-8") as f:
            first_lines.append(f.readline().strip())

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(first_lines) + "\n")

# A6: Extract first H1 titles from markdown files
def index_markdown():
    docs_dir = os.path.join(DATA_DIR, "docs")
    output_file = os.path.join(DATA_DIR, "docs/index.json")
    enforce_security(output_file)

    index = {}
    for file in os.listdir(docs_dir):
        if file.endswith(".md"):
            file_path = os.path.join(docs_dir, file)
            enforce_security(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("# "):
                        index[file] = line[2:].strip()
                        break

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

# A7: Extract sender's email from email.txt
def email_sender():
    input_file = os.path.join(DATA_DIR, "email.txt")
    output_file = os.path.join(DATA_DIR, "email-sender.txt")
    enforce_security(input_file)
    enforce_security(output_file)

    email_pattern = r"From:.*?<(.*?)>"
    extracted_email = ""

    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            match = re.search(email_pattern, f.read().strip())
            extracted_email = match.group(1) if match else ""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_email + "\n")

# A8: Extract credit card number from image using Tesseract
def extract_credit_card():
    input_image = os.path.join(DATA_DIR, "credit-card.png")
    output_file = os.path.join(DATA_DIR, "credit-card.txt")
    enforce_security(input_image)
    enforce_security(output_file)
    
    image = Image.open(input_image)
    text = pytesseract.image_to_string(image)
    card_pattern = r"\b(?:\d[ -]*?){13,16}\b"
    match = re.search(card_pattern, text)
    extracted_card_number = re.sub(r"\D", "", match.group()) if match else ""
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_card_number + "\n")

# A9: Find most similar comments
def similar_comments():
    input_file = os.path.join(DATA_DIR, "comments.txt")
    output_file = os.path.join(DATA_DIR, "comments-similar.txt")
    enforce_security(input_file)
    enforce_security(output_file)

    with open(input_file, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]

    if len(comments) < 2:
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(comments, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    np.fill_diagonal(similarities, 0)
    max_idx = np.unravel_index(np.argmax(similarities), similarities.shape)
    comment1, comment2 = comments[max_idx[0]], comments[max_idx[1]]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{comment1}\n{comment2}\n")

# A10: Calculate total sales for 'Gold' tickets
def total_gold_sales():
    db_file = os.path.join(DATA_DIR, "ticket-sales.db")
    output_file = os.path.join(DATA_DIR, "ticket-sales-gold.txt")
    enforce_security(db_file)
    enforce_security(output_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0] or 0
    conn.close()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{total_sales}\n")

@app.post("/run")
def run_task(task: str):
    task_map = {
        "count_wednesdays": count_wednesdays,
        "sort_contacts": sort_contacts,
        "recent_logs": recent_logs,
        "index_markdown": index_markdown,
        "email_sender": email_sender,
        "extract_credit_card": extract_credit_card,
        "similar_comments": similar_comments,
        "total_gold_sales": total_gold_sales,
    }
    if task not in task_map:
        raise HTTPException(status_code=400, detail="Invalid task")
    try:
        task_map[task]()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get")
def get_task_output(task: str):
    output_map = {
        "count_wednesdays": "dates-wednesdays.txt",
        "sort_contacts": "contacts-sorted.json",
        "recent_logs": "logs-recent.txt",
        "index_markdown": "docs/index.json",
        "email_sender": "email-sender.txt",
        "extract_credit_card": "credit-card.txt",
        "similar_comments": "comments-similar.txt",
        "total_gold_sales": "ticket-sales-gold.txt",
    }
    if task not in output_map:
        raise HTTPException(status_code=400, detail="Invalid task")
    try:
        output_file = os.path.join(DATA_DIR, output_map[task])
        return {"output": read_output_file(output_file)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
