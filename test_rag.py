from src.ingest_enhanced import parse_pdf_enhanced

chunks = parse_pdf_enhanced('data/raw/Seymour_2016.pdf')
for c in chunks:
    if c['two_column'] and c['page'] >= 3:
        print(f"Page {c['page']}:")
        print(c['text'][:800])
        break