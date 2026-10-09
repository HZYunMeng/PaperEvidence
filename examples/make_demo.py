"""Create an original fictional paper for offline smoke tests; not research data."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


def build(out=None):
    out = Path(out or Path(__file__).parent / "demo-paper.pdf")
    styles = getSampleStyleSheet()
    story = []
    def paragraph(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 12))
    paragraph("Fictional Retrieval Study", "Title")
    paragraph("SYNTHETIC DEMO DOCUMENT. All methods, datasets and results below are fictional. Not a scientific publication.")
    paragraph("Abstract", "Heading1")
    paragraph("We illustrate evidence-aware paper retrieval using a fictional dataset named DemoQA. This document is for software testing only.")
    paragraph("1 Introduction", "Heading1")
    paragraph("Readers need to inspect the evidence behind paper summaries. Our demonstration preserves page numbers and retrieves text and ruled tables.")
    paragraph("2 Methods", "Heading1")
    paragraph("The proposed method uses section-aware chunking. The synthetic training set contains 120 examples. The baseline uses fixed character windows.")
    paragraph("The evaluation metric is exact-match accuracy. Retrieval uses BM25 in this demo. No model was trained to produce these fictional results.")
    story.append(PageBreak())
    paragraph("3 Experiments", "Heading1")
    paragraph("The proposed method achieves an accuracy of 91.2% on DemoQA. The baseline accuracy is 82.4%. These values are invented fixtures, not evidence of real improvement.")
    paragraph("Table 1. Fictional results on DemoQA.")
    table = Table([["Method", "Accuracy", "Latency"], ["Baseline", "82.4%", "240 ms"], ["Proposed", "91.2%", "310 ms"]], colWidths=[150, 110, 110])
    table.setStyle(TableStyle([("GRID", (0,0),(-1,-1),1,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7edf4")), ("BOTTOMPADDING",(0,0),(-1,-1),10), ("TOPPADDING",(0,0),(-1,-1),10)]))
    story.append(table)
    story.append(Spacer(1, 20))
    paragraph("4 Limitations", "Heading1")
    paragraph("The demo does not include scanned documents or visual reasoning. The retrieval baseline cannot match cross-language paraphrases reliably.")
    paragraph("5 Conclusion", "Heading1")
    paragraph("This fixture supports testing of page provenance, table preservation and retrieval behavior. Real paper evaluation is required before making quality claims.")
    SimpleDocTemplate(str(out), pagesize=A4, rightMargin=54, leftMargin=54, topMargin=48, bottomMargin=48).build(story)
    return out


if __name__ == "__main__":
    print(build())
