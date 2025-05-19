from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

def makeReport(folder: str):
    report = SimpleDocTemplate(f"{folder}/final/report.pdf", pagesize = A4,
    leftMargin=cm * 1,
    rightMargin=cm * 1,
    topMargin=cm * 1,
    bottomMargin=cm * 1) # Make A4 page
    styles = getSampleStyleSheet() # Sample styles
    docElements = [] # Array to store items in report
    notes = open(f"{folder}/final/note.txt", "r")

    # Title
    docElements.append(Paragraph("Performance Report", styles["Title"]))

    # Starting Sanitation
    docElements.append(Paragraph("Starting Sanitation Performance", styles["Heading2"]))
    docElements.append(Paragraph("Sanitation Coverage", styles["Heading3"]))
    docElements.append(Image(f"{folder}/process/0.png", width=12*cm, height=9*cm))
    note = notes.readline()
    docElements.append(Paragraph(f"{note}", styles["Normal"]))
    docElements.append(Spacer(1, 1.2 * cm))
    
    # Material Gathering
    docElements.append(Paragraph("Material Collection", styles["Heading2"]))
    docElements.append(Paragraph("Material Location", styles["Heading3"]))
    docElements.append(Image(f"{folder}/process/2.png", width=12*cm, height=9*cm))
    docElements.append(Paragraph("Gathered Materials", styles["Heading3"]))
    note = notes.readline()
    splitNote = note.replace('\t',"<br/>")
    docElements.append(Paragraph(f"{splitNote}", styles["Normal"]))
    # docElements.append(Spacer(1, 1.2 * cm))
    
    # Tool Use
    docElements.append(Paragraph("Tool Usage", styles["Heading2"]))
    note = notes.readline()
    splitNote = note.replace('\t',"<br/>")
    docElements.append(Paragraph(f"{splitNote}", styles["Normal"]))
    docElements.append(Spacer(1, 1.2 * cm))

    # Clean Up
    docElements.append(Paragraph("Workspace Cleanup", styles["Heading2"]))
    note = notes.readline()
    docElements.append(Paragraph(f"{note}", styles["Normal"]))
    docElements.append(Spacer(1, 3 * cm))
    
    # Closing Sanitation
    docElements.append(Paragraph("Closing Sanitation Performance", styles["Heading2"]))
    docElements.append(Image(f"{folder}/process/4.png", width=12*cm, height=9*cm))
    note = notes.readline()
    docElements.append(Paragraph(f"{note}", styles["Normal"]))
    docElements.append(Spacer(1, 1.2 * cm))

    report.build(docElements)

if __name__=="__main__":
    makeReport(f"video/140525 010243")