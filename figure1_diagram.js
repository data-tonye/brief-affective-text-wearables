const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
const W = 10, H = 10.0;
pres.defineLayout({ name: 'FIG', width: W, height: H });
pres.layout = 'FIG';
const s = pres.addSlide();
s.background = { color: 'FFFFFF' };
const F = 'Arial';
const BLUE = { fill: 'E3EAF4', line: '3F5F8F', ink: '1F2F4A' };
const ORANGE = { fill: 'FDEBDD', line: 'D9743F', ink: '9A4318' };
const GREEN = { fill: 'E2F1E6', line: '3B8A55', ink: '1F5C36' };
const GOLD = { fill: 'FCF3DF', line: 'CE9A2E', ink: '8A6100' };
const INK = '1F2F4A';

function box(x, y, w, h, c, runs, opt = {}) {
  s.addText(runs, Object.assign({
    shape: pres.shapes.RECTANGLE, x, y, w, h, fill: { color: c.fill }, line: { color: c.line, width: 1.5 },
    fontFace: F, align: 'center', valign: 'middle', margin: 0.06, isTextBox: true,
  }, opt));
}
const t = (text, o = {}) => ({ text, options: Object.assign({ fontSize: 12, color: INK }, o) });
const nl = (text, o = {}) => t(text, Object.assign({ breakLine: true }, o));
function arrow(x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipV: y2 < y1, flipH: x2 < x1, line: { color: '3F5F8F', width: 1.75, endArrowType: 'triangle' } });
}

// Source study
box(1.7, 0.15, 6.6, 0.95, BLUE, [
  nl('SOURCE STUDY', { fontSize: 10.5, color: '55657F' }),
  nl('LEMURS Study', { fontSize: 15, bold: true }),
  t('University of Vermont · 2023–2024 academic year', { fontSize: 10.5 }),
]);
// Row B
box(0.3, 1.5, 2.7, 1.2, BLUE, [
  nl('Enrolled', { fontSize: 11.5 }), nl('N = 487 participants', { fontSize: 14, bold: true }), t('Oura ring fitted at baseline', { fontSize: 11 })]);
box(3.65, 1.5, 2.7, 1.2, ORANGE, [
  nl('Excluded', { fontSize: 11.5, color: ORANGE.ink }), nl('n = 29 participants', { fontSize: 14, bold: true, color: ORANGE.ink }),
  t('Insufficient wear: >50% of activity days with non-wear >240 min/day', { fontSize: 10.5, color: ORANGE.ink })]);
box(7.0, 1.5, 2.7, 1.2, GREEN, [
  nl('Analytic Sample', { fontSize: 11.5, color: GREEN.ink }), nl('N = 458 participants', { fontSize: 14, bold: true, color: GREEN.ink }),
  t('3,610 person-waves · Weeks 1–33', { fontSize: 11, color: GREEN.ink })]);
arrow(5.0, 1.1, 5.0, 1.5);
arrow(3.0, 2.1, 3.65, 2.1);
arrow(6.35, 2.1, 7.0, 2.1);

// Row C
box(0.3, 3.15, 4.7, 2.45, BLUE, [
  nl('BI-MONTHLY CONCERN TEXT (active)', { fontSize: 12.5, bold: true }),
  nl('"Of all the things that have happened in the last 2 weeks, what concerns you the most?"', { fontSize: 11, italic: true, color: '2C4E86' }),
  nl('3,610 total waves · 3,073 concern-present (85.1%)', { fontSize: 11 }),
  nl('Concern-present responses: median 4 words · mean 6.9 (SD 9.7) · range 1–157', { fontSize: 11 }),
  t('Single-word responses 21.1% · Most common: money, school, finals, exams, grades', { fontSize: 11 }),
], { paraSpaceAfter: 3 });
s.addShape(pres.shapes.LINE, { x: 5.18, y: 3.15, w: 0, h: 2.45, line: { color: '8A94A6', width: 1.25, dashType: 'dash' } });
box(5.35, 3.15, 4.35, 2.45, BLUE, [nl('OURA RING (passive continuous sensing)', { fontSize: 12.5, bold: true })], { valign: 'top', margin: 0.1 });
s.addText([nl('Sleep (6 outcomes)', { fontSize: 11.5, bold: true }),
  nl('· Duration (hrs)', { fontSize: 11 }), nl('· Efficiency (%)', { fontSize: 11 }), nl('· REM sleep (hrs)', { fontSize: 11 }),
  nl('· Deep sleep (hrs)', { fontSize: 11 }), nl('· Onset latency (min)', { fontSize: 11 }), t('· RMSSD (ms)', { fontSize: 11 })],
  { x: 5.45, y: 3.62, w: 2.15, h: 1.55, fontFace: F, color: INK, align: 'left', valign: 'top', margin: 0.03, isTextBox: true });
s.addText([nl('Physical activity (3 outcomes)', { fontSize: 11.5, bold: true }),
  nl('· Steps/day', { fontSize: 11 }), nl('· MET-min medium', { fontSize: 11 }), t('· MET-min high', { fontSize: 11 })],
  { x: 7.65, y: 3.62, w: 2.0, h: 1.55, fontFace: F, color: INK, align: 'left', valign: 'top', margin: 0.03, isTextBox: true });
s.addText('Weekly aggregation · Nightly/daily means', { x: 5.45, y: 5.2, w: 4.15, h: 0.3, fontFace: F, fontSize: 11, italic: true, color: INK, align: 'center', margin: 0, isTextBox: true });
arrow(1.65, 2.7, 1.65, 3.15);
arrow(8.35, 2.7, 8.35, 3.15);

// NLP block
box(0.3, 6.0, 6.6, 2.5, GOLD, [nl('NLP FEATURE EXTRACTION', { fontSize: 12.5, bold: true, color: GOLD.ink })], { valign: 'top', margin: 0.08, fill: { color: 'FCF3DF' } });
const mw = 2.05, mg = 0.1, mx = 0.4;
[['SEANCE', '27 dictionary features', 'All 3,610 waves'], ['RoBERTa-base', '127 PCs (PCA)', 'Concern-present only'], ['MentalRoBERTa', '100 PCs (PCA)', 'Concern-present only']].forEach((m, i) => {
  box(mx + i * (mw + mg), 6.4, mw, 1.0, { fill: 'FFFFFF', line: GOLD.line, ink: GOLD.ink }, [
    nl(m[0], { fontSize: 12.5, bold: true, color: GOLD.ink }), nl(m[1], { fontSize: 11.5 }), t(m[2], { fontSize: 11.5 })]);
});
box(0.4, 7.5, 6.4, 0.9, { fill: 'FFFFFF', line: GOLD.line }, [
  nl('Zero-shot domain classification', { fontSize: 12.5, bold: true, color: GOLD.ink }),
  t('8 domains × 9 outcomes = 72 tests · Concern-present only', { fontSize: 11.5 })]);
arrow(2.65, 5.6, 2.65, 6.0);

// Models
box(0.3, 8.8, 9.4, 1.0, BLUE, [
  nl('WITHIN-PERSON MIXED-EFFECTS MODELS', { fontSize: 12.5, bold: true }),
  nl('Participant random intercepts · Week as a fixed effect', { fontSize: 11.5, italic: true }),
  t('Embedding and domain models: N = 2,982 sleep / 2,966 activity concern-present person-waves · ICC range 0.53–0.91', { fontSize: 10.5 })]);
arrow(2.65, 8.5, 2.65, 8.8);
arrow(7.5, 5.6, 7.5, 8.8);
pres.writeFile({ fileName: 'outputs/figures/figure1.pptx' }).then(() => console.log('ok'));
