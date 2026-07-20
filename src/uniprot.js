/**
 * UniProtKB REST helpers (browser → https://rest.uniprot.org).
 * Search by accession, entry name (ID), gene, or protein name.
 */

const BASE = "https://rest.uniprot.org/uniprotkb";
const SEARCH_FIELDS = [
  "accession",
  "id",
  "protein_name",
  "gene_names",
  "organism_name",
  "length",
  "reviewed",
].join(",");
const ENTRY_FIELDS = [
  "accession",
  "id",
  "protein_name",
  "gene_names",
  "organism_name",
  "length",
  "sequence",
  "cc_function",
  "ft_variant",
  "ft_mutagen",
  "protein_existence",
  "annotation_score",
].join(",");

/** Accession like P04637 / A0A024R1R8 */
const ACCESSION_RE = /^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$/i;
/** Entry name like P53_HUMAN */
const ENTRY_ID_RE = /^[A-Z0-9]{1,10}_[A-Z0-9]{1,15}$/i;

function escapeSolrTerm(raw) {
  return String(raw || "")
    .trim()
    .replace(/[+\-&|!(){}[\]^"~*?:\\/]/g, "\\$&");
}

/**
 * Build a UniProt query that matches Entry, Entry Name, gene, or protein name.
 */
export function buildSearchQuery(raw) {
  const q = String(raw || "").trim();
  if (!q) return "";
  if (ACCESSION_RE.test(q)) {
    const a = q.toUpperCase();
    return `accession:${a} OR id:${a}`;
  }
  if (ENTRY_ID_RE.test(q)) {
    return `id:${q.toUpperCase()}`;
  }
  const term = escapeSolrTerm(q);
  // Prefer gene / protein name / id; also allow free-text fallback
  return `(gene:${term}) OR (protein_name:${term}) OR (id:${term}) OR (${term})`;
}

async function fetchJson(url) {
  const res = await fetch(url, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `UniProt ${res.status}${body ? `: ${body.slice(0, 160)}` : ""}`,
    );
  }
  return res.json();
}

function proteinNameFromDescription(desc) {
  if (!desc) return "";
  const rec = desc.recommendedName?.fullName?.value;
  if (rec) return rec;
  const alt = desc.alternativeNames?.[0]?.fullName?.value;
  return alt || "";
}

function geneNamesFromEntry(genes) {
  if (!Array.isArray(genes) || !genes.length) return { primary: "", all: [] };
  const primary = genes[0]?.geneName?.value || "";
  const all = [];
  for (const g of genes) {
    if (g.geneName?.value) all.push(g.geneName.value);
    for (const s of g.synonyms || []) {
      if (s.value) all.push(s.value);
    }
  }
  return { primary, all: [...new Set(all)] };
}

function functionText(comments) {
  const fn = (comments || []).find((c) => c.commentType === "FUNCTION");
  const texts = fn?.texts || [];
  return texts.map((t) => t.value).filter(Boolean).join(" ");
}

function locStart(feature) {
  const v = feature?.location?.start?.value;
  return typeof v === "number" ? v : null;
}

function locEnd(feature) {
  const v = feature?.location?.end?.value;
  return typeof v === "number" ? v : locStart(feature);
}

function parseVariantFeature(feature, kind) {
  const start = locStart(feature);
  const end = locEnd(feature);
  const alt = feature.alternativeSequence || {};
  const original = alt.originalSequence || "";
  const alternatives = Array.isArray(alt.alternativeSequences)
    ? alt.alternativeSequences.filter((x) => typeof x === "string")
    : [];
  // Empty alternative list often means deletion
  const toList = alternatives.length ? alternatives : [""];
  const changes = toList.map((to) => {
    const from = original || "?";
    const label =
      start == null
        ? `${from}→${to || "Δ"}`
        : start === end
          ? `${from}${start}${to || "Δ"}`
          : `${from}${start}-${end}${to || "Δ"}`;
    return { from, to, label };
  });
  return {
    kind,
    id: feature.featureId || "",
    start,
    end,
    description: feature.description || "",
    original,
    alternatives: toList,
    changes,
    label: changes[0]?.label || "",
  };
}

function parseFeatures(features) {
  const natural = [];
  const mutant = [];
  for (const f of features || []) {
    if (f.type === "Natural variant") {
      natural.push(parseVariantFeature(f, "natural"));
    } else if (f.type === "Mutagenesis") {
      mutant.push(parseVariantFeature(f, "mutant"));
    }
  }
  natural.sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
  mutant.sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
  return { natural, mutant };
}

export function summarizeSearchHit(raw) {
  const genes = geneNamesFromEntry(raw.genes);
  return {
    accession: raw.primaryAccession || "",
    entryName: raw.uniProtkbId || "",
    proteinName: proteinNameFromDescription(raw.proteinDescription),
    gene: genes.primary,
    genes: genes.all,
    organism:
      raw.organism?.commonName ||
      raw.organism?.scientificName ||
      "",
    scientificName: raw.organism?.scientificName || "",
    length:
      raw.sequence?.length ||
      (typeof raw.sequence === "number" ? raw.sequence : null),
    reviewed: String(raw.entryType || "").toLowerCase().includes("reviewed"),
    entryType: raw.entryType || "",
  };
}

export function summarizeEntry(raw) {
  const base = summarizeSearchHit(raw);
  const seq = raw.sequence?.value || "";
  const { natural, mutant } = parseFeatures(raw.features);
  return {
    ...base,
    length: raw.sequence?.length || seq.length || base.length,
    sequence: seq,
    functionText: functionText(raw.comments),
    proteinExistence: raw.proteinExistence || "",
    annotationScore: raw.annotationScore ?? null,
    natural,
    mutant,
    uniprotUrl: base.accession
      ? `https://www.uniprot.org/uniprotkb/${base.accession}`
      : "",
  };
}

/**
 * Search UniProtKB. Returns up to `size` hit summaries.
 */
export async function searchUniProt(rawQuery, size = 12) {
  const query = buildSearchQuery(rawQuery);
  if (!query) throw new Error("Enter an accession, entry name, gene, or protein name.");
  const params = new URLSearchParams({
    query,
    fields: SEARCH_FIELDS,
    format: "json",
    size: String(size),
  });
  const data = await fetchJson(`${BASE}/search?${params}`);
  const results = (data.results || []).map(summarizeSearchHit);
  const total = Number(
    // header may be missing when calling via some proxies; body has no total
    results.length,
  );
  return { results, query, total };
}

/**
 * Fetch one entry with sequence + natural variants + mutagenesis.
 */
export async function fetchUniProtEntry(accession) {
  const acc = String(accession || "").trim().toUpperCase();
  if (!acc) throw new Error("Missing UniProt accession.");
  const params = new URLSearchParams({ fields: ENTRY_FIELDS });
  const data = await fetchJson(`${BASE}/${encodeURIComponent(acc)}?${params}`);
  return summarizeEntry(data);
}

/**
 * Apply a UniProt variant/mutagenesis change onto a wild-type sequence.
 * Position is 1-based UniProt numbering.
 */
export function applySequenceEdit(sequence, variant, altIndex = 0) {
  const seq = String(sequence || "");
  if (!variant || variant.start == null) {
    throw new Error("Variant has no coordinates.");
  }
  const start = variant.start - 1;
  const end = (variant.end ?? variant.start) - 1;
  const fromLen = Math.max(1, end - start + 1);
  const expected = variant.original || seq.slice(start, start + fromLen);
  const to = variant.alternatives?.[altIndex] ?? "";
  if (start < 0 || start >= seq.length) {
    throw new Error(`Position ${variant.start} is outside the sequence.`);
  }
  const observed = seq.slice(start, start + expected.length || fromLen);
  if (expected && observed && observed !== expected) {
    // Soft warning — still apply at coordinates (isoform / version drift)
    console.warn(
      `UniProt variant ${variant.label}: expected ${expected} at ${variant.start}, found ${observed}`,
    );
  }
  const replaceLen = expected ? expected.length : fromLen;
  return seq.slice(0, start) + to + seq.slice(start + replaceLen);
}

export function filterVariants(list, rawFilter) {
  const q = String(rawFilter || "").trim().toLowerCase();
  if (!q) return list || [];

  // Pure residue number → exact / spanning position (avoid matching dbSNP ids)
  if (/^\d+$/.test(q)) {
    const pos = Number(q);
    return (list || []).filter((v) => {
      if (v.start == null) return false;
      const end = v.end ?? v.start;
      return pos >= v.start && pos <= end;
    });
  }

  const range = q.match(/^(\d+)\s*[-–:]\s*(\d+)$/);
  if (range) {
    const lo = Math.min(Number(range[1]), Number(range[2]));
    const hi = Math.max(Number(range[1]), Number(range[2]));
    return (list || []).filter((v) => {
      if (v.start == null) return false;
      const end = v.end ?? v.start;
      return v.start <= hi && end >= lo;
    });
  }

  const motif = q.toUpperCase().replace(/[^A-Z0-9]/g, "");
  return (list || []).filter((v) => {
    if (v.label?.toLowerCase().includes(q)) return true;
    if (v.id?.toLowerCase().includes(q)) return true;
    if (motif && v.label?.toUpperCase().includes(motif)) return true;
    if (v.description?.toLowerCase().includes(q)) return true;
    return false;
  });
}
