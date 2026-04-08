/**
 * Generate submission-ready .docx for Paper 1.
 * Uses docx-js with ACL-inspired academic formatting.
 *
 * Usage: node paper/generate_docx.js
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, LevelFormat,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  ExternalHyperlink, FootnoteReferenceRun,
} = require("docx");

const FIGURES = path.join(__dirname, "figures");
const OUT = path.join(__dirname, "Cost-Aware_Hybrid_Routing_Paper.docx");

// ─── Style constants ────────────────────────────────────────────
const FONT = "Times New Roman";
const FONT_SZ = 22; // 11pt
const SMALL = 20;   // 10pt
const TINY = 18;    // 9pt
const H1_SZ = 28;   // 14pt
const H2_SZ = 24;   // 12pt
const TITLE_SZ = 36; // 18pt

const BLUE = "0072B2";
const RED  = "D55E00";
const GRAY = "666666";
const LIGHT_BLUE = "E8F4FD";
const LIGHT_GRAY = "F5F5F5";

// ─── Helper functions ───────────────────────────────────────────
function p(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    // Parse **bold** and *italic*
    const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
    for (const part of parts) {
      if (part.startsWith("**") && part.endsWith("**")) {
        runs.push(new TextRun({ text: part.slice(2, -2), bold: true, font: FONT, size: opts.size || FONT_SZ }));
      } else if (part.startsWith("*") && part.endsWith("*")) {
        runs.push(new TextRun({ text: part.slice(1, -1), italics: true, font: FONT, size: opts.size || FONT_SZ }));
      } else if (part) {
        runs.push(new TextRun({ text: part, font: FONT, size: opts.size || FONT_SZ, ...opts.run }));
      }
    }
  } else {
    runs.push(...(Array.isArray(text) ? text : [text]));
  }
  return new Paragraph({
    children: runs,
    spacing: { after: opts.after ?? 120, before: opts.before ?? 0, line: opts.line ?? 276 },
    alignment: opts.align ?? AlignmentType.JUSTIFIED,
    indent: opts.indent,
    ...(opts.heading ? { heading: opts.heading } : {}),
  });
}

function heading(level, text) {
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
    children: [new TextRun({
      text,
      font: FONT,
      size: level === 1 ? H1_SZ : H2_SZ,
      bold: true,
    })],
    spacing: { before: level === 1 ? 360 : 240, after: 120 },
  });
}

function caption(text) {
  return p(text, { size: SMALL, align: AlignmentType.CENTER, after: 200, run: { italics: true, color: GRAY } });
}

function figure(filename, w, h, capText) {
  const imgPath = path.join(FIGURES, filename);
  const data = fs.readFileSync(imgPath);
  return [
    new Paragraph({
      children: [new ImageRun({
        type: "png",
        data,
        transformation: { width: w, height: h },
        altText: { title: filename, description: capText, name: filename },
      })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 80 },
    }),
    caption(capText),
  ];
}

// ─── Table helpers ──────────────────────────────────────────────
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
const TABLE_W = 9360; // full content width

function tableCell(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    for (const part of parts) {
      if (part.startsWith("**") && part.endsWith("**")) {
        runs.push(new TextRun({ text: part.slice(2, -2), bold: true, font: FONT, size: TINY }));
      } else if (part) {
        runs.push(new TextRun({ text: part, font: FONT, size: TINY, ...(opts.bold ? { bold: true } : {}) }));
      }
    }
  } else {
    runs.push(new TextRun({ text: String(text), font: FONT, size: TINY, ...(opts.bold ? { bold: true } : {}) }));
  }
  return new TableCell({
    borders,
    width: { size: opts.width || 1000, type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    shading: opts.header ? { fill: LIGHT_BLUE, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ children: runs, alignment: opts.align ?? AlignmentType.LEFT })],
  });
}

function tableRow(cells, opts = {}) {
  return new TableRow({ children: cells.map((c, i) => tableCell(c, { ...opts, width: opts.widths?.[i] })) });
}

// ═══════════════════════════════════════════════════════════════
// DOCUMENT
// ═══════════════════════════════════════════════════════════════
async function main() {
  const doc = new Document({
    styles: {
      default: { document: { run: { font: FONT, size: FONT_SZ } } },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: H1_SZ, bold: true, font: FONT },
          paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 },
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: H2_SZ, bold: true, font: FONT },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 },
        },
      ],
    },
    sections: [{
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: { top: 1440, right: 1296, bottom: 1440, left: 1296 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            children: [new TextRun({ text: "Cost-Aware Hybrid Routing for Intent Classification", font: FONT, size: TINY, italics: true, color: GRAY })],
            alignment: AlignmentType.CENTER,
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            children: [new TextRun({ text: "Page ", font: FONT, size: TINY, color: GRAY }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: TINY, color: GRAY })],
            alignment: AlignmentType.CENTER,
          })],
        }),
      },
      children: [
        // ─── TITLE ───
        new Paragraph({
          children: [new TextRun({ text: "Cost-Aware Hybrid Routing for Intent Classification:", font: FONT, size: TITLE_SZ, bold: true })],
          alignment: AlignmentType.CENTER,
          spacing: { before: 200, after: 0 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "74% LLM Cost Reduction at Equal Accuracy", font: FONT, size: TITLE_SZ, bold: true })],
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 200 },
        }),
        // ─── AUTHOR ───
        new Paragraph({
          children: [new TextRun({ text: "Bo-Yu Chen", font: FONT, size: FONT_SZ, bold: true })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 40 },
        }),
        new Paragraph({
          children: [
            new TextRun({ text: "University of Texas at San Antonio", font: FONT, size: SMALL, italics: true }),
          ],
          alignment: AlignmentType.CENTER,
          spacing: { after: 40 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "boyu.chen@my.utsa.edu", font: FONT, size: SMALL, color: BLUE })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
        }),

        // ─── ABSTRACT ───
        heading(1, "Abstract"),
        p("Routing user utterances to the correct task agent is a foundational step in multi-agent LLM systems, and the LLM-as-router pattern has become the default for its accuracy. We ask whether that accuracy is worth its cost. On CLINC150 (150 intents mapped to 7 agents + OOS), we compare four routers spanning a 3-order-of-magnitude cost range: (R1) keyword, (R2) embedding-nearest-centroid, (R3) Claude Haiku 4.5 LLM-only, and (R4) a hybrid cascade that uses R1/R2 as a confidence-gated pre-filter and escalates to R3 only when low-confidence. Across 3 seeds of 400 stratified queries (1,200 pooled), R4 achieves **82.6% \u00b1 1.2pp** accuracy versus R3's **82.9% \u00b1 0.6pp** \u2014 statistically indistinguishable by McNemar paired test (p > 0.37) \u2014 while escalating to the LLM on only **26.1%** of queries, yielding a **74.3% reduction in LLM cost**. We release code, tuned thresholds, and per-seed trajectories."),

        // ─── 1. INTRODUCTION ───
        heading(1, "1. Introduction"),
        p("In production multi-agent LLM systems, every user utterance first visits a *router*: a component that decides which downstream agent \u2014 finance, HR, travel, and so on \u2014 should handle the request. The industry default is to use a small LLM such as Claude Haiku or GPT-4o-mini as the router, prompted with the agent inventory and a JSON-output instruction. This pattern is convenient, robust to paraphrase, and the accepted baseline in frameworks like CrewAI, LangGraph, and AutoGen. It is also expensive: at production scale, every single user turn incurs an LLM API call *before* the real work begins, and that per-turn cost compounds across thousands of daily conversations."),
        p("The question this paper asks is whether that per-turn LLM call is actually necessary. A keyword match or an embedding lookup against per-agent centroids costs three to four orders of magnitude less than an API call, runs on local compute with millisecond latency, and \u2014 on easy queries \u2014 can be just as accurate as the LLM. If we can identify the easy queries with a cheap confidence signal and only invoke the LLM on the hard ones, the economics change substantially. We quantify this on CLINC150, a standard 150-intent benchmark, by comparing four routers across a three-order-of-magnitude cost spectrum and measuring the accuracy\u2013cost Pareto frontier."),
        p("Our contributions are four-fold. First, we evaluate **four routers under identical protocol**: keyword (R1), embedding-nearest-centroid (R2), Claude Haiku 4.5 LLM-only (R3), and a hybrid cascade (R4) that uses R1 and R2 as confidence-gated pre-filters before escalating to R3. Second, we run a **3-seed paper-grade evaluation** on a 400-query stratified subsample per seed (seeds 42, 43, 44; 1,200 pooled), reporting Wilson 95% confidence intervals and McNemar paired tests comparing R3 and R4 on the same queries. Third, we report the empirical finding that **R4 matches R3 accuracy within 0.4pp** (82.6% vs 82.9%) while escalating to the LLM on only 26% of queries \u2014 a **74.3% reduction in LLM cost** with no statistically significant accuracy loss at \u03b1=0.05. Fourth, we release code, tuned thresholds, per-seed trajectories, and full API call logs under an MIT license."),
        p("The remainder of the paper is organised as follows. \u00a72 positions this work against intent-classification and LLM-cascade literature. \u00a73 describes the four routers and the threshold-tuning protocol. \u00a74 presents 3-seed results with significance tests and qualitative examples. \u00a75 discusses when cascades pay off, limitations, and the extension to multi-turn agent tasks that we pursue in companion work."),

        // ─── 2. RELATED WORK ───
        heading(1, "2. Related Work"),
        p("**Intent classification on CLINC150.** The CLINC150 benchmark (Larson et al. 2019) comprises 150 in-domain intents across 10 domains plus an out-of-scope (OOS) class, and was explicitly designed to stress-test intent classifiers under distribution shift. Strong baselines include SetFit (Tunstall et al. 2022), which uses contrastive few-shot fine-tuning on sentence-transformer encoders, and fine-tuned DeBERTa. We use SetFit as a no-LLM baseline in our headline table (70.2% on the full test set), and Claude Haiku 4.5 zero-shot as our cost ceiling."),
        p("**Cost-aware LLM cascades.** FrugalGPT (Chen et al. 2023) introduced the general cascade pattern \u2014 try a cheaper model first, escalate to a more expensive model only when the cheap model\u2019s confidence is low \u2014 on single-turn question answering tasks, reporting cost reductions of up to 98% at matched accuracy. Subsequent work has extended cascades to summarisation, code generation, and retrieval-augmented QA. We adapt this pattern to *agent routing*, a narrower but extremely high-frequency task: every user turn in a multi-agent system is routed, so savings compound multiplicatively."),
        p("**Agent-routing systems.** Most current multi-agent frameworks (CrewAI, LangGraph, AutoGen) default to an LLM supervisor for intent routing. Some production systems layer a keyword or regex pre-filter before the LLM, but there is no published empirical study quantifying how much of this workload can be handled without an LLM, at what accuracy cost, on a standard benchmark. Our contribution is not a new routing algorithm; it is the empirical answer \u2014 on CLINC150 \u2014 to the question \u201cwhat fraction of routing decisions can a keyword or embedding handle, and what does the accuracy\u2013cost Pareto curve look like?\u201d"),
        p("**Position.** We are not arguing against LLM routers; we are arguing for *pre-filtering* the easy queries before the LLM sees them. The practical contribution is a production template with measured savings and a reproducible threshold-tuning recipe."),

        // ─── 3. METHOD ───
        heading(1, "3. Method"),
        heading(2, "3.1 Dataset and Label Space"),
        p("We use the CLINC150 test split (5,500 queries, including 1,000 OOS queries). Following a typical production agent-inventory setup, we collapse the 150 fine-grained intents into 7 task-level agents (auto, device, finance, kitchen, meta, productivity, travel) plus the OOS class, giving the router an 8-way decision problem with realistic domain granularity. Class sizes after collapsing range from 360 to 1,140 queries."),

        heading(2, "3.2 The Four Routers"),
        p("**R1 \u2014 Keyword router.** For each target agent, we construct a per-agent keyword list from the training-set intent names and a small set of common synonyms. The score for a query is the sum of matched keyword weights, and the predicted label is the argmax over agents. We report the raw keyword score as R1\u2019s confidence."),
        p("**R2 \u2014 Embedding router.** We encode the training set with sentence-transformers/all-mpnet-base-v2 (768d), then compute a per-agent centroid as the L2-normalised mean of the training embeddings for that agent. At inference, a query\u2019s embedding is compared to all centroids by cosine similarity, and the nearest-centroid agent is returned. R2\u2019s confidence is the cosine similarity to the top-1 centroid."),
        p("**R3 \u2014 LLM router.** Claude Haiku 4.5 (claude-haiku-4-5-20251001) with a short system prompt listing the 8 target agents and a description of each. The model is asked to return a single JSON object. Temperature is 0. Each query is a single API call with no in-context examples."),
        p("**R4 \u2014 Hybrid cascade.** R4 is a confidence-gated cascade over R1 \u2192 R2 \u2192 R3 with two thresholds (kt, et). If R1_confidence \u2265 kt, return R1\u2019s prediction; else if R2_confidence \u2265 et, return R2\u2019s prediction; otherwise call R3 (LLM). We refer to this full cascade as **R4+LLM**. We also report **R4 no-LLM** \u2014 the same cascade with the LLM escalation branch replaced by R2\u2019s best guess \u2014 as a fair no-API baseline."),

        heading(2, "3.3 Threshold Tuning"),
        p("Thresholds are tuned on a held-out validation split via grid search over kt \u2208 {0.5, 1.0, 1.5, 2.0} and et \u2208 {0.05, 0.08, 0.10, 0.15}. The objective maximises accuracy \u2212 0.1 \u00b7 LLM_call_rate, which prefers configurations that trade LLM calls for cheap-stage calls at equal accuracy. The tuned values for R4+LLM are kt = 0.5, et = 0.10."),

        heading(2, "3.4 Evaluation Protocol"),
        p("LLM evaluation runs use a per-seed stratified subsample of n = 400 (50 queries per class \u00d7 8 classes) to bound API spend. We use 3 seeds (42, 43, 44) for paper-grade variance estimation. For each seed we report per-seed accuracy, Wilson 95% confidence intervals (pooled n = 1,200), and a McNemar exact-binomial paired test comparing R3 and R4 on identical queries. Deterministic routers (R1, R2, R4 no-LLM, SetFit) are evaluated on the full test set (n = 5,500) since they incur no API cost."),

        heading(2, "3.5 Cost Accounting"),
        p("R3 and R4 API costs are measured directly from Anthropic response metadata (input + output tokens \u00d7 Haiku 4.5 pricing). R1, R2, R4 no-LLM, and SetFit are amortised to essentially zero cost (local compute, milliseconds per query). The total API cost for the full 3-seed experiment was **$0.44** \u2014 $0.35 for R3 (1,200 queries \u00d7 3 seeds) and $0.09 for R4+LLM."),

        // ─── 4. RESULTS ───
        heading(1, "4. Results"),

        heading(2, "4.1 Main Results"),
        p("Table 1 shows the headline numbers across all routers.", { after: 80 }),

        // Table 1 — Headline
        p("**Table 1.** Main results on CLINC150. Bold rows are the primary comparison (LLM-based routers).", { size: SMALL, after: 60, run: { italics: true } }),
        new Table({
          width: { size: TABLE_W, type: WidthType.DXA },
          columnWidths: [2800, 1200, 1500, 1400, 1100, 1360],
          rows: [
            tableRow(["Router", "Acc.", "Wilson CI", "LLM rate", "Cost/q"], { header: true, widths: [2800, 1200, 1500, 1400, 1100, 1360] }),
            tableRow(["R1 keyword (n=5,500)", "64.8%", "\u2014", "0%", "~$0"], { widths: [2800, 1200, 1500, 1400, 1100, 1360] }),
            tableRow(["R2 embedding (n=5,500)", "74.0%", "\u2014", "0%", "~$0"], { widths: [2800, 1200, 1500, 1400, 1100, 1360] }),
            tableRow(["SetFit baseline (n=5,500)", "70.2%", "\u2014", "0%", "~$0"], { widths: [2800, 1200, 1500, 1400, 1100, 1360] }),
            tableRow(["R4 no-LLM (n=5,500)", "74.9%", "\u2014", "0%", "~$0"], { widths: [2800, 1200, 1500, 1400, 1100, 1360] }),
            tableRow(["**R3 LLM-only (n=1,200)**", "**82.9%**", "[80.7, 84.9]", "100%", "$2.93\u00d710\u207b\u2074"], { widths: [2800, 1200, 1500, 1400, 1100, 1360] }),
            tableRow(["**R4 hybrid+LLM (n=1,200)**", "**82.6%**", "[80.3, 84.6]", "26.1%", "$7.52\u00d710\u207b\u2075"], { widths: [2800, 1200, 1500, 1400, 1100, 1360] }),
          ],
        }),
        p(""),

        // Figure 1
        ...figure("F1_accuracy_vs_llm_rate.png", 520, 380,
          "Figure 1. Cost-accuracy trade-off on CLINC150. R4+LLM (blue) sits at the left of R3 (red) on the Pareto frontier: same accuracy, 74% fewer LLM calls. Gray dots are baselines."),

        heading(2, "4.2 Statistical Significance"),
        p("McNemar paired exact-binomial tests on R3 vs R4+LLM, per seed (n = 400 each):", { after: 80 }),

        // Table 2 — McNemar
        p("**Table 2.** McNemar paired tests per seed.", { size: SMALL, after: 60, run: { italics: true } }),
        new Table({
          width: { size: TABLE_W, type: WidthType.DXA },
          columnWidths: [1800, 2500, 2500, 2560],
          rows: [
            tableRow(["Seed", "\u0394 accuracy (R4\u2212R3)", "McNemar p", "Sig. at \u03b1=0.05"], { header: true, widths: [1800, 2500, 2500, 2560] }),
            tableRow(["42", "+0.00 pp", "1.000", "No"], { widths: [1800, 2500, 2500, 2560] }),
            tableRow(["43", "\u22121.75 pp", "0.371", "No"], { widths: [1800, 2500, 2500, 2560] }),
            tableRow(["44", "+0.75 pp", "0.755", "No"], { widths: [1800, 2500, 2500, 2560] }),
          ],
        }),
        p(""),
        p("Across all three seeds the difference is not significant. R4+LLM is statistically indistinguishable from R3."),

        // Figure 2
        ...figure("F2_per_seed_bars.png", 520, 360,
          "Figure 2. Per-seed comparison. Each seed\u2019s R3 and R4+LLM score on the same 400 queries; connecting lines show paired deltas. R4 loses on seed 43 (\u22121.8pp) and wins slightly on 44 (+0.8pp)."),

        heading(2, "4.3 Cost Savings"),
        p("Per-seed, R3 LLM-only costs $0.117 in API usage (400 calls \u00d7 Haiku 4.5 pricing), while R4+LLM costs $0.030 (\u2248105 escalated calls). The mean LLM call rate for R4+LLM across seeds is 26.1% \u00b1 1.7pp, yielding an overall **LLM cost reduction of 74.3%** against R3. Projected to 1,000 queries, R4+LLM costs $0.075 versus R3\u2019s $0.293 \u2014 a saving of $0.22 per 1,000 queries that scales linearly."),

        // Figure 4
        ...figure("F4_cost_bars.png", 520, 300,
          "Figure 3. LLM API cost per 1,000 queries. R4 hybrid+LLM ($0.075) is 74% cheaper than R3 LLM-only ($0.293) at statistically equivalent accuracy."),

        heading(2, "4.4 Qualitative Analysis"),
        p("The three cascade stages partition the seed-42 query distribution as: **57.5% stop at R1** (88.7% accurate), **17.0% stop at R2** (75.0% accurate), **25.5% escalate to R3**. Two patterns dominate R1\u2019s 11.3% error rate: polysemous tokens (\u201ccall\u201d triggers productivity instead of device, \u201cvisa\u201d triggers travel instead of finance) and OOS leakage (the keyword router cannot refuse to classify). The cascade correctly escalates 64% of OOS queries to the LLM versus 10\u201318% for most in-domain agents."),

        // Table 3 — Per-agent
        p("**Table 3.** Escalation rate and R1-stop error rate by agent (seed 42, n=400).", { size: SMALL, after: 60, run: { italics: true } }),
        new Table({
          width: { size: TABLE_W, type: WidthType.DXA },
          columnWidths: [2000, 2000, 2200, 3160],
          rows: [
            tableRow(["Agent", "Escalation", "R1-stop error", "Interpretation"], { header: true, widths: [2000, 2000, 2200, 3160] }),
            tableRow(["travel", "10.0%", "9.5%", "Easy: domain verbs"], { widths: [2000, 2000, 2200, 3160] }),
            tableRow(["auto", "14.0%", "0.0%", "Easy: keyword anchors"], { widths: [2000, 2000, 2200, 3160] }),
            tableRow(["finance", "16.0%", "4.8%", "Easy: financial nouns"], { widths: [2000, 2000, 2200, 3160] }),
            tableRow(["kitchen", "18.0%", "7.1%", "Mostly easy"], { widths: [2000, 2000, 2200, 3160] }),
            tableRow(["productivity", "18.0%", "11.8%", "Moderate: shared verbs"], { widths: [2000, 2000, 2200, 3160] }),
            tableRow(["device", "22.0%", "25.0%", "Hard: polysemous commands"], { widths: [2000, 2000, 2200, 3160] }),
            tableRow(["meta", "42.0%", "13.3%", "Hard: conversational"], { widths: [2000, 2000, 2200, 3160] }),
            tableRow(["**OOS**", "**64.0%**", "**100.0%**", "**Hardest: correctly escalated**"], { widths: [2000, 2000, 2200, 3160] }),
          ],
        }),
        p(""),

        // Figure 3
        ...figure("F3_per_agent_heatmap.png", 560, 310,
          "Figure 4. Per-agent accuracy heatmap across all routers (top 5\u00d78) with delta row (R4+LLM \u2212 R3). R4 matches or beats R3 on 7/8 agents; the OOS loss (\u221216pp) is the expected trade-off."),

        // ─── 5. DISCUSSION ───
        heading(1, "5. Discussion"),
        p("**When does the cascade pay off?** A cascade wins when the query distribution is *bimodal* \u2014 many easy queries answerable from surface features plus a tail of hard queries that need an LLM\u2019s open-ended reasoning. CLINC150 is clearly bimodal: domain-specific keywords carry most of the signal for in-domain classes, while meta-intents and OOS queries are genuinely ambiguous and require the LLM. A benchmark with flat difficulty would show less benefit."),
        p("**When does it fail?** If the cheap router\u2019s confidence calibration is poor, the cascade loses accuracy without saving cost. The R1 false-confidence zone illustrated in the qualitative analysis (polysemous tokens) is the primary failure mode on CLINC150. Our objective function accuracy \u2212 0.1 \u00b7 LLM_call_rate penalises this miscalibration. In production, the 0.1 coefficient should be replaced with the real LLM cost per query normalised by the product\u2019s accuracy tolerance."),
        p("**Limitations.** (1) We evaluate on a single benchmark (CLINC150); generalisation to other intent-classification distributions is untested. (2) The 8-way agent inventory is fixed; cascades on much larger label spaces (50+ agents) would require a different cheap-stage design. (3) The thresholds are static \u2014 no online adaptation. (4) Haiku 4.5 is itself a cheap model; savings against Claude Sonnet or GPT-4 would be larger in absolute dollars. (5) We measure cost in API dollars, not user-perceived latency."),
        p("**Extension: multi-turn agent tasks.** The cascade pattern extends naturally to multi-turn settings. Instead of routing every turn, one can use a cheap model for task *decomposition* and save the expensive model for *execution*. We evaluate this on \u03c4-bench (Sierra 2024) in companion work."),

        // ─── 6. CONCLUSION ───
        heading(1, "6. Conclusion"),
        p("On CLINC150, a confidence-gated hybrid cascade matches an LLM-only router within 0.4pp accuracy while reducing LLM calls by 74%. The design choice is not \u201cLLM-as-router or not\u201d \u2014 it is \u201cLLM-as-router for *which* queries.\u201d Production teams paying per-turn API costs on high-frequency routing should have a template for measuring and tuning this trade-off on their own label space; we release ours as that template."),

        // ─── REPRODUCIBILITY ───
        heading(1, "Reproducibility Statement"),
        p("Code, tuned thresholds, per-seed trajectories, LLM call logs: github.com/drewOrc/cost-aware-hybrid-router (MIT). Model: claude-haiku-4-5-20251001. Seeds: 42, 43, 44. Embedding: sentence-transformers/all-mpnet-base-v2. Total API cost to reproduce: **$0.44**."),

        // ─── REFERENCES ───
        heading(1, "References"),
        p("Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. *arXiv preprint arXiv:2305.05176*.", { size: SMALL, after: 80 }),
        p("Larson, S., Mahendran, A., Peper, J. J., Clarke, C., Lee, A., Hill, P., Kummerfeld, J. K., Leach, K., Laurenzano, M. A., Tang, L., & Mars, J. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. *Proceedings of EMNLP-IJCNLP 2019*, 1311\u20131316.", { size: SMALL, after: 80 }),
        p("Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP 2019*.", { size: SMALL, after: 80 }),
        p("Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D., Wasserblat, M., & Pereg, O. (2022). Efficient Few-Shot Learning Without Prompts. *arXiv preprint arXiv:2209.11055*.", { size: SMALL, after: 80 }),
        p("Anthropic. (2025). Claude Haiku 4.5 \u2014 Model Card and Pricing. docs.anthropic.com/en/docs/about-claude/models. Accessed 2026-04-05.", { size: SMALL, after: 80 }),
      ],
    }],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUT, buffer);
  console.log(`\u2713 Written: ${OUT}`);
  console.log(`  Size: ${(buffer.length / 1024).toFixed(0)} KB`);
}

main().catch(console.error);
