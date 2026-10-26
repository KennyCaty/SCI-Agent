def format_segment(title: str, heading: str, text: str) -> str:
    fields = [f"Title: {title.strip() or '(untitled)'}"]
    if heading.strip() and heading.strip().casefold() != title.strip().casefold():
        fields.append(f"Section: {heading.strip()}")
    fields.append(f"Passage: {text.strip()}")
    return '\n'.join(fields)
