"""
Build complete PFE report: figures, screenshots, DOCX, PDF.
"""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(BASE, "tools")
VENV_PY = os.path.join(BASE, ".venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PY):
    VENV_PY = sys.executable

DOCX_OUT = os.path.join(BASE, "Rapport_PFE_Raddaoui_Wael_Master_DS_IA.docx")
DOCX_FALLBACK_OUT = os.path.join(BASE, "Rapport_PFE_Raddaoui_Wael_Master_DS_IA_mobile.docx")
PDF_OUT = os.path.join(BASE, "Rapport_PFE_Raddaoui_Wael_Master_DS_IA.pdf")


def run(cmd, cwd=None):
    print(f"\n>> {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd or BASE)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def export_pdf(docx_path: str, pdf_path: str):
    try:
        from docx2pdf import convert
        convert(docx_path, pdf_path)
        print(f"PDF (docx2pdf): {pdf_path}")
        return
    except Exception as e:
        print(f"docx2pdf failed: {e}")

    try:
        import comtypes.client
        word = comtypes.client.CreateObject("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(docx_path))
        doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17)
        doc.Close()
        word.Quit()
        print(f"PDF (Word COM): {pdf_path}")
        return
    except Exception as e:
        print(f"Word COM failed: {e}")

    soffice = r"C:\Program Files\LibreOffice\program\soffice.exe"
    if os.path.exists(soffice):
        run([
            soffice, "--headless", "--convert-to", "pdf",
            "--outdir", os.path.dirname(pdf_path), docx_path,
        ])
        print(f"PDF (LibreOffice): check {pdf_path}")
        return

    print("ERROR: Could not export PDF. Install Microsoft Word or: pip install docx2pdf")


def main():
    run([VENV_PY, os.path.join(TOOLS, "generate_all_figures.py")])

    try:
        run([VENV_PY, os.path.join(TOOLS, "capture_screenshots.py")])
    except SystemExit:
        print("Screenshots skipped or failed — continuing with existing images.")

    run([VENV_PY, os.path.join(BASE, "generate_rapport.py")])

    docx_path = DOCX_OUT if os.path.exists(DOCX_OUT) else DOCX_FALLBACK_OUT
    if os.path.exists(docx_path):
        export_pdf(docx_path, PDF_OUT)
    else:
        print("DOCX not found:", DOCX_OUT)

    print("\n=== BUILD COMPLETE ===")
    print("DOCX:", docx_path if os.path.exists(docx_path) else DOCX_OUT)
    if os.path.exists(PDF_OUT):
        print("PDF :", PDF_OUT)


if __name__ == "__main__":
    main()
