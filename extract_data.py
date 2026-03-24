import os
import json
import re
from pathlib import Path

# Try importing the necessary libraries to read PDFs and Word docs
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("Warning: PyPDF2 is missing. Install with 'pip install PyPDF2' to read PDFs.")
    PdfReader = None

try:
    import docx
except ImportError:
    print("Warning: python-docx is missing. Install with 'pip install python-docx' to read DOCX files.")
    docx = None

# Set the target dataset directory
BASE_DIR = "/Users/luke/Downloads/docs/Filemail.com - eduquest"
# Output file for our structured data
OUTPUT_JSON = os.path.join(BASE_DIR, "extracted_syllabus_data.json")

def extract_metadata_from_filename(filename):
    """
    Tries to infer the Level, Subject, and Term from the given filename.
    """
    filename_lower = filename.lower()
    metadata = {
        "level": "Unknown",
        "subject": "Unknown",
        "term": "Unknown"
    }

    # 1. Infer Level (P1-P7, Baby, Middle, Top)
    level_match = re.search(r'p\s*\.?\s*([1-7])', filename_lower)
    if level_match:
        metadata["level"] = f"P{level_match.group(1)}"
    elif "primary one" in filename_lower: metadata["level"] = "P1"
    elif "primary two" in filename_lower: metadata["level"] = "P2"
    elif "primary three" in filename_lower: metadata["level"] = "P3"
    elif "primary four" in filename_lower: metadata["level"] = "P4"
    elif "primary five" in filename_lower: metadata["level"] = "P5"
    elif "primary six" in filename_lower: metadata["level"] = "P6"
    elif "primary seven" in filename_lower: metadata["level"] = "P7"
    elif "baby" in filename_lower: metadata["level"] = "Baby Class"
    elif "middle" in filename_lower: metadata["level"] = "Middle Class"
    elif "top" in filename_lower: metadata["level"] = "Top Class"
    elif "nursery" in filename_lower: metadata["level"] = "Nursery"

    # 2. Infer Subject
    if re.search(r'\b(maths?|mathematics|mtc)\b', filename_lower):
        metadata["subject"] = "Mathematics"
    elif re.search(r'\b(eng|english)\b', filename_lower):
        metadata["subject"] = "English"
    elif re.search(r'\b(sci|scie|science)\b', filename_lower):
        metadata["subject"] = "Science"
    elif re.search(r'\b(sst|social\s*studies)\b', filename_lower):
        metadata["subject"] = "Social Studies"
    elif re.search(r'\b(re|c\.?r\.?e|i\.?r\.?e|religious)\b', filename_lower):
        metadata["subject"] = "Religious Education"
    elif r"la" in filename_lower or "literacy" in filename_lower:
        metadata["subject"] = "Literacy"

    # 3. Infer Term
    term_match = re.search(r'term\s*([1-3]|i{1,3}|one|two|three)', filename_lower)
    if term_match:
        term_val = term_match.group(1)
        if term_val in ['1', 'i', 'one']: metadata["term"] = "Term 1"
        elif term_val in ['2', 'ii', 'two']: metadata["term"] = "Term 2"
        elif term_val in ['3', 'iii', 'three']: metadata["term"] = "Term 3"
    
    # 4. Infer Document Type
    if "scheme" in filename_lower: metadata["doc_type"] = "Scheme of Work"
    elif "mock" in filename_lower or "exam" in filename_lower or "ple" in filename_lower or "eot" in filename_lower or "bot" in filename_lower: 
        metadata["doc_type"] = "Exam / Past Paper"
    elif "note" in filename_lower or "breakdown" in filename_lower: 
        metadata["doc_type"] = "Lesson Notes"
    else:
        metadata["doc_type"] = "Other"

    return metadata


def extract_text_from_pdf(filepath):
    text = ""
    if not PdfReader:
        return text
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
    return text


def extract_text_from_docx(filepath):
    text = ""
    if not docx:
        return text
    try:
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {filepath}: {e}")
    return text


def process_directory(directory):
    processed_data = []
    
    for root, dirs, files in os.walk(directory):
        # Skip zip files or non-document files
        for file in files:
            filepath = os.path.join(root, file)
            ext = file.lower().split('.')[-1]
            
            if ext not in ['pdf', 'docx', 'doc']:
                continue
                
            print(f"Processing: {file}")
            
            # 1. Metadata Tagging
            metadata = extract_metadata_from_filename(file)
            metadata['filename'] = file
            metadata['filepath'] = filepath
            
            # 2. Text Extraction
            extracted_text = ""
            if ext == 'pdf':
                extracted_text = extract_text_from_pdf(filepath)
            elif ext == 'docx':
                extracted_text = extract_text_from_docx(filepath)
            elif ext == 'doc':
                # Note: older .doc files might need 'antiword' or specific libraries.
                # We skip deep extraction for .doc files here to prevent crashing
                extracted_text = "[Notice: .doc format requires antiword or conversion to .docx to extract text reliably.]"
            
            # Only add to our dataset if we successfully extracted some text
            if len(extracted_text.strip()) > 50 or ext == 'doc': # Allow saving the metadata even if text failed for .doc
                metadata['content_length'] = len(extracted_text)
                # To prevent gigantic JSONs initially, we can take a chunk or just snippet. 
                # Later, we will store the full text directly into a Vector DB.
                metadata['content'] = extracted_text 
                
                processed_data.append(metadata)

    return processed_data


if __name__ == "__main__":
    print(f"Starting Data Extraction Pipeline on {BASE_DIR}...")
    dataset = process_directory(BASE_DIR)
    
    print(f"\nExtraction complete! Processed {len(dataset)} documents.")
    
    # Save to JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Data successfully saved to {OUTPUT_JSON}")
