def load_docs():
    import json
    import pathlib
    import pandas as pd
    path = pathlib.Path("../../")
    spells = json.loads((path / "spells_pdf.json").read_text())
    weapons = json.loads((path / "weapons.json").read_text())
    feats = json.loads((path / "feats_pdf.json").read_text())
    classes = json.loads((path / "classes_pdf.json").read_text())

    def format_value(v):
        if isinstance(v, bool): return "yes" if v else "no"
        elif isinstance(v, list): return ", ".join(str(i) for i in v)
        return str(v)

    def clean_record(rec: dict, rec_type: str = "record") -> str:
        if "desc" in rec and "description" not in rec:
            rec["description"] = rec.pop("desc")
        priority_keys = [
            "name", "description", "school", "level", "spell_level", "level_int", "page",
            "casting_time", "range", "components", "material", "duration", "concentration",
            "ritual", "dnd_class", "spell_lists", "archetype", "circles"
        ]
        exclude_keys = {
            "slug", "document__url", "document__title", "document__license_url",
            "target_range_sort", "document__slug"
        }
        lines = [f"type: {rec_type}"]
        for key in priority_keys:
            if key in rec and key not in exclude_keys and rec[key] not in (None, ""):
                lines.append(f"{key}: {format_value(rec[key])}")
        for key, val in rec.items():
            if key not in priority_keys and key not in exclude_keys and val not in (None, ""):
                lines.append(f"{key}: {format_value(val)}")
        return "\n".join(lines)

    return (
        [clean_record(s, "spell")   for s in spells] +
        [clean_record(w, "weapon")  for w in weapons] +
        [clean_record(f, "feat")    for f in feats] +
        [clean_record(c, "class")   for c in classes]
    )


if __name__ == "__main__":
    print("✅ Testing load_documents.py...")
    
    docs = load_docs()  # <--- THIS LINE is critical

    print(f"Total docs: {len(docs)}\n")
    print("🧪 Sample document:")
    print(docs[0][:500])

