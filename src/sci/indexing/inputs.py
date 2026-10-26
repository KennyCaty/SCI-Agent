from sci.indexing.common import DATA, INDEX, MODEL, ROOT, config, save, sha256


def collect():
    manifest={}
    for root in [MODEL,INDEX/'source_bm25',DATA/'ner_cache']:
        for path in sorted(root.rglob('*')):
            if path.is_file():manifest[str(path.relative_to(ROOT))]=sha256(path)
    for path in [DATA/'sources.jsonl',DATA/'headings.sqlite',ROOT/'src/sci/indexing/segment_format.py']:
        manifest[str(path.relative_to(ROOT))]=sha256(path)
    record={'configuration':config(),'file_sha256':manifest}
    save(ROOT/'outputs/indexing/reports/BUILD_INPUTS.json',record)
    return record


if __name__=='__main__':collect()
