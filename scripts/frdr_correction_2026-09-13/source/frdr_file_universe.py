"""Complete FRDR evaluation annotations over exactly the manifest file universe."""


def complete_annotations(manifest, annotations):
    files = list(manifest["base"].unique())
    outside = set(annotations) - set(files)
    if outside:
        raise ValueError("Annotation files absent from manifest: " + ", ".join(sorted(outside)))
    return {base: annotations.get(base, []) for base in files}
