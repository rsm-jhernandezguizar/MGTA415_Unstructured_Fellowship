def load_docs():
    import json
    import pathlib

    path = pathlib.Path("../../")

    # Load JSON files
    spells = json.loads((path / "spells_pdf.json").read_text())
    weapons = json.loads((path / "weapons.json").read_text())
    feats = json.loads((path / "feats_pdf.json").read_text())
    classes = json.loads((path / "classes_pdf.json").read_text())

    # Optional: print file sizes loaded
    print(f"🔍 Loaded counts:")
    print(f"  Spells:  {len(spells)}")
    print(f"  Weapons: {len(weapons)}")
    print(f"  Feats:   {len(feats)}")
    print(f"  Classes: {len(classes)}\n")

    # Converts fields to strings
    def format_value(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        elif isinstance(v, list):
            return ", ".join(str(i) for i in v)
        return str(v)

    # Clean one record
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

    # Apply cleaner with error logging
    def try_clean(batch, label):
        results = []
        errors = 0
        for rec in batch:
            try:
                result = clean_record(rec, label)
                if result:
                    results.append(result)
            except Exception as e:
                errors += 1
                print(f"❌ Skipped {label}: {rec.get('name', '[no name]')} — {e}")
        print(f"✅ {label.title()}: {len(results)} records loaded, {errors} skipped\n")
        return results

    # Clean and return all data
    return (
        try_clean(spells, "spell") +
        try_clean(weapons, "weapon") +
        try_clean(feats, "feat") +
        try_clean(classes, "class")
    )


# Test the function if run directly
if __name__ == "__main__":
    print("✅ Testing load_documents.py...\n")
    docs = load_docs()
    print(f"\n📦 Total cleaned records: {len(docs)}")
    print("🧪 Sample record:\n")
    print(docs[0] if docs else "⚠️ No documents returned.")
