#!/usr/bin/env python3
"""
Script to create a sample PDF from the medical report text
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import os

def create_medical_report_pdf():
    """Create a PDF file from the sample medical report"""
    
    # Read the sample medical report text
    with open('sample_medical_report.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate("sample_medical_report.pdf", pagesize=letter)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor='#2c3e50'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor='#34495e'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        leading=14
    )
    
    # Split content into sections
    sections = content.split('\n\n')
    
    # Add title
    story.append(Paragraph("MEDICAL REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Process each section
    for section in sections:
        if section.strip():
            lines = section.strip().split('\n')
            
            # Check if this is a heading (all caps or ends with colon)
            first_line = lines[0].strip()
            if first_line.isupper() or first_line.endswith(':'):
                # This is a heading
                story.append(Paragraph(first_line, heading_style))
                
                # Add remaining lines as normal text
                for line in lines[1:]:
                    if line.strip():
                        story.append(Paragraph(line.strip(), normal_style))
            else:
                # This is normal text
                for line in lines:
                    if line.strip():
                        story.append(Paragraph(line.strip(), normal_style))
            
            story.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(story)
    
    print("✅ Sample medical report PDF created successfully!")
    print("📄 File: sample_medical_report.pdf")
    print("📁 Location: " + os.path.abspath("sample_medical_report.pdf"))
    print("\n🚀 You can now upload this PDF to test the Medical Report RAG Extractor!")

if __name__ == "__main__":
    try:
        create_medical_report_pdf()
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        print("Make sure you have the sample_medical_report.txt file in the same directory.")
