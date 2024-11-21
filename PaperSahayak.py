import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import spacy
from tkinter import Tk, filedialog, Text, Scrollbar, messagebox, StringVar, Label, Entry
import tkinter as tk
import customtkinter as ctk
from nltk.stem import WordNetLemmatizer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
from threading import Thread

# --- PDF Analyzer Setup ---
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOPWORDS = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def extract_text_from_pdf(pdf_path):
    import pdfplumber
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def save_text_to_file(text, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(text)

def extract_keywords(query):
    doc = nlp(query)
    lemmatizer = WordNetLemmatizer()
    keywords = {lemmatizer.lemmatize(token.text.lower()) for token in doc if token.is_alpha and not token.is_stop}
    return list(keywords)

def search_keywords_in_file(file_path, keywords, context_window=2):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    sentences = sent_tokenize(text)
    relevant_sentences = []
    for i, sentence in enumerate(sentences):
        for keyword in keywords:
            if re.search(rf'\b{keyword}\b', sentence, re.IGNORECASE):
                start = max(0, i - context_window)
                end = min(len(sentences), i + context_window + 1)
                relevant_sentences.extend(sentences[start:end])
    return list(dict.fromkeys(relevant_sentences))

def compile_context(sentences):
    return " ".join(sentences)

def process_context_for_answer(context, min_words=50, max_words=100):
    """Process the context and generate a meaningful descriptive answer."""
    # Clean up excessive spaces and split into sentences
    context = context.strip()
    context = re.sub(r'\s+', ' ', context)  # Clean up excessive spaces
    
    # Focus on relevant details and clean out irrelevant information (like citations, metadata)
    context_points = sent_tokenize(context)
    
    # Remove metadata-like sentences that aren't directly related to the topic
    filtered_context = [sent for sent in context_points if len(sent.split()) > 5 and not re.search(r'\d{4}', sent)]

    # Select the most relevant part for brevity and coherence
    processed_context = " ".join(filtered_context[:3])  # Select the first few meaningful sentences
    
    # Make sure the context doesn't exceed max words (we limit for better focus)
    processed_context_words = processed_context.split()
    if len(processed_context_words) > max_words:
        processed_context = " ".join(processed_context_words[:max_words]) + "..."

    return processed_context

def generate_detailed_answer(question, context, min_words=50, max_words=150):
    """Generate a detailed descriptive answer within the word limit, utilizing processed context."""
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']

    # Process context for relevant details
    processed_context = process_context_for_answer(context, max_words)

    # Combine the answer and processed context
    detailed_answer = (
        f"{answer.capitalize()}. "
        f"{processed_context}."
    )

    # Ensure the answer fits within the word limit
    words = detailed_answer.split()
    if len(words) < min_words:
        additional_context = context.split()[:min_words - len(words)]
        detailed_answer += " " + " ".join(additional_context)
    elif len(words) > max_words:
        detailed_answer = " ".join(words[:max_words]) + "..."

    # Remove unfinished sentences by ensuring the context ends cleanly
    detailed_answer = re.sub(r'\s+[A-Za-z0-9]*$', '', detailed_answer)  # Trims incomplete words/sentences

    return detailed_answer

def answer_question(question, extracted_text_file, keywords):
    """Answer a question based on dynamically generated context from the extracted text."""
    relevant_sentences = search_keywords_in_file(extracted_text_file, keywords, context_window=5)
    context = compile_context(relevant_sentences)

    if not context.strip():
        return "No relevant information found in the text."

    try:
        # Generate a descriptive AI answer
        detailed_answer = generate_detailed_answer(question, context)

        # Return the detailed AI-generated answer
        return detailed_answer
    except Exception as e:
        return f"Error processing question: {e}"

# --- Web Scraper Setup ---
def fetch_paper_titles(query):
    # Setup WebDriver with headless option
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU (for headless compatibility)
    chrome_options.add_argument("--no-sandbox")  # Prevent sandboxing (may be needed for some systems)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    results = []

    # List of URLs to scrape
    urls = [
        f"https://ieeexplore.ieee.org/Xplore/home.jsp?queryText={query}",
        f"https://scholar.google.com/scholar?q={query}",
        f"https://www.researchgate.net/search.Search.html?query={query}",
        f"https://sci-hub.se/{query}"
    ]

    try:
        for url in urls:
            driver.get(url)
            time.sleep(3)  # Wait for the page to load

            # For Google Scholar
            if "scholar.google" in url:
                titles = driver.find_elements(By.CSS_SELECTOR, "h3.gs_rt a")
                for title in titles:
                    results.append(f"Title: {title.text}\nLink: {title.get_attribute('href')}\n")

            # For IEEE Xplore
            elif "ieeexplore.ieee.org" in url:
                titles = driver.find_elements(By.CSS_SELECTOR, "h3 a")
                for title in titles:
                    results.append(f"Title: {title.text}\nLink: {title.get_attribute('href')}\n")

            # For ResearchGate
            elif "researchgate.net" in url:
                titles = driver.find_elements(By.CSS_SELECTOR, "a.cite-title span")
                links = driver.find_elements(By.CSS_SELECTOR, "a.cite-title")
                for title, link in zip(titles, links):
                    results.append(f"Title: {title.text}\nLink: {link.get_attribute('href')}\n")

            # For Sci-Hub
            elif "sci-hub.se" in url:
                titles = driver.find_elements(By.CSS_SELECTOR, "h3.title a")
                for title in titles:
                    results.append(f"Title: {title.text}\nLink: {title.get_attribute('href')}\n")

    finally:
        driver.quit()

    return results

# --- GUI Setup ---
def upload_pdf():
    file_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])
    if file_path:
        text = extract_text_from_pdf(file_path)
        save_text_to_file(text, "extracted_text.txt")
        messagebox.showinfo("Success", "PDF uploaded and text extracted!")
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Text saved to 'extracted_text.txt'.\n")

def ask_question():
    question = question_entry.get()
    if not os.path.exists("extracted_text.txt"):
        messagebox.showerror("Error", "No extracted text file found. Upload a PDF first.")
        return

    # Extract keywords from the question
    keywords = extract_keywords(question)

    if not keywords:
        messagebox.showwarning("No Keywords", "No keywords could be extracted from the question.")
        return

    # Call answer_question with the extracted keywords
    answer = answer_question(question, "extracted_text.txt", keywords)

    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, f"Q: {question}\nA: {answer}\n")


def scrape_web():
    query = web_query_entry.get()
    if not query.strip():
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    def scrape_and_display():
        results = fetch_paper_titles(query)
        if results:
            output_text = "\n".join(results)
        else:
            output_text = "No results found for the given query."

        # Update the output box in the main thread
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, output_text)

    # Run the scraping in a separate thread to avoid freezing the GUI
    Thread(target=scrape_and_display).start()

# --- Main GUI Layout ---
root = tk.Tk()
root.title("PDF & Web Scraper")
root.geometry("1280x720")
root.configure(bg="#2b2b2b")
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# PDF Section
pdf_frame = ctk.CTkFrame(root)
pdf_frame.pack(pady=10, padx=10, fill="x")
ctk.CTkLabel(pdf_frame, text="Research Paper Analyzer", font=("Helvetica", 24)).pack(pady=5)
ctk.CTkButton(pdf_frame, text="Upload PDF", command=upload_pdf).pack(pady=5)
question_entry = ctk.CTkEntry(pdf_frame, placeholder_text="Ask a question about the PDF...")
question_entry.pack(pady=5, padx=10, fill="x")
ctk.CTkButton(pdf_frame, text="Get Answer", command=ask_question).pack(pady=5)

# Web Scraper Section
web_frame = ctk.CTkFrame(root)
web_frame.pack(pady=10, padx=10, fill="x")
ctk.CTkLabel(web_frame, text="Research Paper Finder", font=("Helvetica", 24)).pack(pady=5)
web_query_entry = ctk.CTkEntry(web_frame, placeholder_text="Enter research paper topic...")
web_query_entry.pack(pady=5, padx=10, fill="x")
ctk.CTkButton(web_frame, text="Find Papers", command=scrape_web).pack(pady=5)

# Output Section
output_frame = ctk.CTkFrame(root)
output_frame.pack(pady=10, padx=10, fill="both", expand=True)
output_box = Text(output_frame, wrap="word")
output_box.pack(pady=10, padx=10, fill="both", expand=True)

root.mainloop()
