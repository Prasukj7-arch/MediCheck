#!/usr/bin/env python3
"""
Simple script to create a PDF from the medical report text using fpdf2
"""

try:
    from fpdf import FPDF
except ImportError:
    print("Installing fpdf2...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

def clean_text(text):
    """Replace special characters with standard ones"""
    replacements = {
        'μ': 'u',  # micro symbol
        '°': ' degrees ',  # degree symbol
        '–': '-',  # en dash
        '—': '-',  # em dash
        '×': 'x',  # multiplication symbol
        '≤': '<=',  # less than or equal
        '≥': '>=',  # greater than or equal
        '±': '+/-',  # plus minus
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def create_medical_report_pdf():
    """Create a PDF file from the sample medical report"""
    
    # Read the sample medical report text
    with open('sample_medical_report.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean the text
    content = clean_text(content)
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Helvetica", size=12)
    
    # Add title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(200, 10, text="MEDICAL REPORT", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)
    
    # Process content
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
            
        # Check if this is a heading (all caps or ends with colon)
        if line.isupper() or line.endswith(':'):
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(200, 8, text=line, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", size=10)
            # Handle long lines by wrapping
            if len(line) > 80:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        pdf.cell(200, 6, text=current_line.strip(), new_x="LMARGIN", new_y="NEXT")
                        current_line = word + " "
                if current_line:
                    pdf.cell(200, 6, text=current_line.strip(), new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.cell(200, 6, text=line, new_x="LMARGIN", new_y="NEXT")
    
    # Save PDF
    pdf.output("sample_medical_report.pdf")
    
    print("✅ Sample medical report PDF created successfully!")
    print("📄 File: sample_medical_report.pdf")
    print("📁 Location: " + os.path.abspath("sample_medical_report.pdf"))
    print("\n🚀 You can now upload this PDF to test the Medical Report RAG Extractor!")

if __name__ == "__main__":
    import os
    try:
        create_medical_report_pdf()
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        print("Make sure you have the sample_medical_report.txt file in the same directory.")
