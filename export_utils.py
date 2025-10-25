
import os
from typing import Optional, List, Dict, Union

def export_markdown(segments: Union[List[Dict], str], summary: Optional[str] = None) -> str:
    """Generate markdown content from segments and optional summary."""
    if isinstance(segments, str):
        # If segments is a string, treat it as raw transcript
        content = f"# Audio Transcript\n\n{segments}\n"
    else:
        # If segments is a list of dictionaries, format them properly
        content = "# Audio Transcript\n\n"
        if segments and len(segments) > 0:
            for segment in segments:
                if 'speaker' in segment:
                    content += f"**{segment['speaker']}** [{segment['start']:.1f}-{segment['end']:.1f}s]: {segment['text']}\n\n"
                else:
                    content += f"{segment['text']}\n\n"
        else:
            content += "*No transcript segments available.*\n\n"
    
    if summary:
        content += f"\n## Summary\n\n{summary}\n"
    
    return content

def export_pdf(segments: Union[List[Dict], str], summary: Optional[str] = None, out_path: str = None) -> Optional[str]:
    """
    Try to export a simple PDF. If reportlab not available, write a .txt file and return its path.
    """
    # Generate content first
    content = export_markdown(segments, summary)
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
        
        doc = SimpleDocTemplate(out_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title = Paragraph("Audio Transcript", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Add content
        if isinstance(segments, str):
            # Raw transcript
            content_para = Paragraph(content.replace('\n', '<br/>'), styles['Normal'])
            story.append(content_para)
        else:
            # Segments with speaker info
            for segment in segments:
                if 'speaker' in segment:
                    speaker_text = f"<b>{segment['speaker']}</b> [{segment['start']:.1f}-{segment['end']:.1f}s]: {segment['text']}"
                else:
                    speaker_text = segment['text']
                para = Paragraph(speaker_text, styles['Normal'])
                story.append(para)
                story.append(Spacer(1, 6))
        
        # Add summary if available
        if summary:
            story.append(Spacer(1, 12))
            summary_title = Paragraph("Summary", styles['Heading2'])
            story.append(summary_title)
            story.append(Spacer(1, 6))
            summary_para = Paragraph(summary.replace('\n', '<br/>'), styles['Normal'])
            story.append(summary_para)
        
        doc.build(story)
        return out_path
    except Exception:
        # fallback to .txt
        txt_path = os.path.splitext(out_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)
        return txt_path
