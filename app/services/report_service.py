import os
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from app.utils.config_loader import load_config
from app.utils.logger import logger

def generate_pdf_report(
    patient_id: str,
    patient_raw: Dict[str, Any],
    probability: float,
    prediction: int,
    top_features: List[Dict[str, Any]],
    model_name: str,
    clinical_explanation: str
) -> str:
    """Generates a professional clinical readmission risk PDF report."""
    config = load_config()
    reports_dir = config["paths"]["reports_dir"]
    pdf_dir = os.path.join(reports_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"risk_report_patient_{patient_id}_{timestamp_str}.pdf"
    pdf_path = os.path.join(pdf_dir, filename)
    
    logger.info(f"Generating PDF report for patient {patient_id} at {pdf_path}...")
    
    # Page setup
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom colors
    primary_color = colors.HexColor("#0f4c81")  # Slate Blue
    secondary_color = colors.HexColor("#2a9d8f")  # Teal
    text_dark = colors.HexColor("#2b2b2b")
    background_light = colors.HexColor("#f8f9fa")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=primary_color,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.gray,
        spaceAfter=20
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=primary_color,
        spaceBefore=15,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_dark,
        leading=14
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBodyDark',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Story flow elements
    story = []
    
    # Title & Header
    story.append(Paragraph("Clinical Readmission Risk Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Platform version: {config['project']['version']}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Grid: Patient Demographics & Prediction Score
    # Left column: Demographic stats
    # Right column: Prediction probability
    risk_level = "HIGH RISK" if probability >= 0.5 else "MEDIUM RISK" if probability >= 0.2 else "LOW RISK"
    risk_color = "#e63946" if probability >= 0.5 else "#f4a261" if probability >= 0.2 else "#2a9d8f"
    
    demo_html = f"""
    <b>Patient ID:</b> {patient_id}<br/>
    <b>Age Midpoint:</b> {patient_raw.get('age_midpoint', 'N/A')} years<br/>
    <b>Race:</b> {patient_raw.get('race', 'Unknown')}<br/>
    <b>Gender:</b> {patient_raw.get('gender', 'N/A')}<br/>
    <b>Time in Hospital:</b> {patient_raw.get('time_in_hospital', 'N/A')} days<br/>
    """
    
    score_html = f"""
    <font size="14"><b>Prediction Status:</b></font><br/>
    <font size="18" color="{risk_color}"><b>{risk_level}</b></font><br/>
    <b>Readmission Probability:</b> {probability * 100:.1f}%<br/>
    <b>Model Used:</b> {model_name.replace('_', ' ')}<br/>
    """
    
    grid_data = [
        [Paragraph(demo_html, body_style), Paragraph(score_html, body_style)]
    ]
    
    grid_table = Table(grid_data, colWidths=[3.5*inch, 3.5*inch])
    grid_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), background_light),
        ('PADDING', (0,0), (-1,-1), 15),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LINEBELOW', (0,0), (-1,-1), 2, primary_color),
    ]))
    
    story.append(grid_table)
    story.append(Spacer(1, 15))
    
    # Section: Key Prediction Drivers (SHAP)
    story.append(Paragraph("Key Readmission Risk Drivers (XAI Analysis)", section_heading))
    story.append(Paragraph("The features below represent the top factors contributing to the patient's predicted readmission risk, measured by SHAP (SHapley Additive exPlanations) values.", body_style))
    story.append(Spacer(1, 8))
    
    # Feature table
    feat_headers = [Paragraph("<b>Risk Feature</b>", bold_body_style), 
                    Paragraph("<b>Patient Value</b>", bold_body_style), 
                    Paragraph("<b>Relative Impact</b>", bold_body_style)]
    
    table_data = [feat_headers]
    for feat in top_features[:5]:
        val_str = f"{feat['value']:.2f}" if isinstance(feat['value'], float) else str(feat['value'])
        impact_str = f"+{feat['shap_value']:.4f}" if feat['shap_value'] > 0 else f"{feat['shap_value']:.4f}"
        
        # Determine cell style color for positive/negative impacts
        impact_color = "red" if feat['shap_value'] > 0 else "green"
        impact_html = f"<font color='{impact_color}'><b>{impact_str}</b></font>"
        
        table_data.append([
            Paragraph(feat['feature'].replace('_', ' ').title(), body_style),
            Paragraph(val_str, body_style),
            Paragraph(impact_html, body_style)
        ])
        
    feat_table = Table(table_data, colWidths=[3.2*inch, 1.8*inch, 2.0*inch])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, background_light]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    # Quick fix for headers color in reportlab (headers need to be whitesmoke text, but bold_body_style specifies text_dark. Let's fix that).
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=bold_body_style,
        textColor=colors.whitesmoke
    )
    for i in range(len(feat_headers)):
        table_data[0][i] = Paragraph(feat_headers[i].text, header_style)
        
    story.append(feat_table)
    story.append(Spacer(1, 15))
    
    # Section: LLM Clinical Interpretation & Guidance
    story.append(Paragraph("Clinical Insights & Actionable Guidance", section_heading))
    
    # Format clinical explanation paragraphs
    paragraphs = clinical_explanation.split("\n\n")
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para.replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 6))
            
    # Page numbers and disclaimer footer
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.gray)
        canvas.drawString(40, 20, "CONFIDENTIAL - For Clinical Use Only. Generated by Healthcare AI Risk Prediction Platform.")
        canvas.drawRightString(letter[0]-40, 20, f"Page {doc.page}")
        canvas.restoreState()
        
    # Build Document
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    logger.info(f"PDF successfully generated.")
    return pdf_path
