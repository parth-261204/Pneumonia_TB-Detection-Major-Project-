import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from datetime import datetime
import json

def generate_diagnostic_report(prediction_result, image_paths, patient_info=None, output_path='reports/diagnostic_report.pdf'):
    """Generate comprehensive diagnostic report with predictions and visualizations"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12
    )
    
    story.append(Paragraph("Chest X-Ray Diagnostic Report", title_style))
    story.append(Paragraph("TB & Pneumonia Detection using Deep Learning", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    if patient_info:
        story.append(Paragraph("Patient Information", heading_style))
        patient_data = [
            ['Patient ID:', patient_info.get('id', 'N/A')],
            ['Date:', patient_info.get('date', datetime.now().strftime('%Y-%m-%d'))],
            ['Age:', patient_info.get('age', 'N/A')],
            ['Gender:', patient_info.get('gender', 'N/A')]
        ]
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ECF0F1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Diagnostic Results", heading_style))
    
    pred_class = prediction_result['predicted_class']
    confidence = prediction_result['confidence'] * 100
    
    color = colors.green if pred_class == 'Normal' else colors.red
    
    diagnosis_text = f"<font color='{color.hexval()}' size='14'><b>Diagnosis: {pred_class}</b></font>"
    story.append(Paragraph(diagnosis_text, styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    confidence_text = f"<b>Confidence Level:</b> {confidence:.2f}%"
    story.append(Paragraph(confidence_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Class Probabilities", heading_style))
    prob_data = [['Class', 'Probability']]
    for cls, prob in prediction_result['probabilities'].items():
        prob_data.append([cls, f"{prob*100:.2f}%"])
    
    prob_table = Table(prob_data, colWidths=[3*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.3*inch))
    
    if 'individual_predictions' in prediction_result:
        story.append(Paragraph("Individual Model Predictions", heading_style))
        model_data = [['Model', 'Prediction', 'Confidence']]
        for model_name, pred in prediction_result['individual_predictions'].items():
            model_data.append([
                model_name.replace('_', ' ').title(),
                pred['class'],
                f"{pred['confidence']*100:.2f}%"
            ])
        
        model_table = Table(model_data, colWidths=[2*inch, 2*inch, 1.5*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ECC71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(model_table)
        story.append(Spacer(1, 0.3*inch))
    
    story.append(PageBreak())
    story.append(Paragraph("Visual Analysis", heading_style))
    
    if os.path.exists(image_paths.get('original', '')):
        story.append(Paragraph("Original X-Ray Image", styles['Heading3']))
        img = Image(image_paths['original'], width=4*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    if os.path.exists(image_paths.get('gradcam', '')):
        story.append(Paragraph("Grad-CAM Heatmap (Attention Map)", styles['Heading3']))
        img = Image(image_paths['gradcam'], width=4*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    if os.path.exists(image_paths.get('overlay', '')):
        story.append(Paragraph("Overlay Visualization", styles['Heading3']))
        img = Image(image_paths['overlay'], width=4*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    story.append(Paragraph("Clinical Recommendations", heading_style))
    
    if pred_class == 'Normal':
        recommendations = [
            "No signs of tuberculosis or pneumonia detected.",
            "Continue routine health monitoring.",
            "Maintain healthy lifestyle practices."
        ]
    elif pred_class == 'Tuberculosis':
        recommendations = [
            "Tuberculosis detected with high confidence.",
            "Immediate consultation with pulmonologist recommended.",
            "Sputum test and culture should be performed.",
            "Start anti-TB treatment protocol if confirmed.",
            "Patient isolation may be necessary to prevent transmission."
        ]
    else:
        recommendations = [
            "Pneumonia detected with high confidence.",
            "Immediate medical attention recommended.",
            "Blood tests and additional imaging may be required.",
            "Antibiotic treatment should be initiated.",
            "Monitor oxygen saturation and respiratory rate."
        ]
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Disclaimer", heading_style))
    disclaimer = """
    This report is generated by an AI-powered diagnostic system and should be used 
    as a supplementary tool only. Final diagnosis must be made by qualified healthcare 
    professionals based on complete clinical evaluation, patient history, and 
    additional diagnostic tests. This system is not a replacement for professional 
    medical judgment.
    """
    story.append(Paragraph(disclaimer, styles['Normal']))
    
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    
    doc.build(story)
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    sample_result = {
        'predicted_class': 'Tuberculosis',
        'confidence': 0.87,
        'probabilities': {
            'Normal': 0.05,
            'Tuberculosis': 0.87,
            'Pneumonia': 0.08
        }
    }
    
    image_paths = {
        'original': 'reports/images/original/input.png',
        'gradcam': 'reports/images/gradcam/heatmap.png',
        'overlay': 'reports/images/overlays/overlay.png'
    }
    
    patient_info = {
        'id': 'PAT-001',
        'date': '2025-12-17',
        'age': '45',
        'gender': 'Male'
    }
    
    generate_diagnostic_report(sample_result, image_paths, patient_info)
