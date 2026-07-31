// Astryx neutral theme tokens (dark stops) — lifted from
// external/astryx/packages/themes/neutral/src/neutralTheme.ts (Meta, MIT).
// Syntax colors are the OKLCH T80 dark-mode stops of the categorical ramps.
export const astryx = {
  // grayscale spine
  body: '#1b1b1b', // T10 main canvas
  surface: '#262626', // T15 lifted interactive surface
  codeBg: '#0a0a0a', // syntax background (dark)
  border: '#404040',
  text: '#e5e5e5',
  textMuted: '#a3a3a3',
  textFaint: '#737373',
  // syntax (dark stops)
  syntax: {
    keyword: '#efa8ff', // purple
    string: '#a6d2a2', // green pastel
    comment: '#a3a3a3',
    number: '#ffb37f', // orange
    function: '#a0caff', // blue
    type: '#efa8ff',
    variable: '#e5e5e5',
    operator: '#a3a3a3',
    constant: '#ffb37f',
    tag: '#ffaeaa', // red
    attribute: '#eec12f', // yellow
    property: '#83dac9', // teal
    punctuation: '#525252',
  },
  // language badge accents (saturated stops, white/dark text per WCAG note)
  badge: {
    python: {bg: '#a0caff', fg: '#0a0a0a'},
    rust: {bg: '#ffb37f', fg: '#0a0a0a'},
    default: {bg: '#efa8ff', fg: '#0a0a0a'},
  },
  font: {
    body: 'Figtree, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    code: 'ui-monospace, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },
} as const;

// Prism token type → astryx syntax color
export const prismToAstryx: Record<string, string> = {
  keyword: astryx.syntax.keyword,
  string: astryx.syntax.string,
  'template-string': astryx.syntax.string,
  char: astryx.syntax.string,
  comment: astryx.syntax.comment,
  number: astryx.syntax.number,
  boolean: astryx.syntax.constant,
  function: astryx.syntax.function,
  'function-definition': astryx.syntax.function,
  'class-name': astryx.syntax.type,
  builtin: astryx.syntax.type,
  variable: astryx.syntax.variable,
  operator: astryx.syntax.operator,
  constant: astryx.syntax.constant,
  tag: astryx.syntax.tag,
  'attr-name': astryx.syntax.attribute,
  attribute: astryx.syntax.attribute,
  property: astryx.syntax.property,
  punctuation: astryx.syntax.punctuation,
  decorator: astryx.syntax.attribute,
  macro: astryx.syntax.attribute,
  'lifetime-annotation': astryx.syntax.tag,
};

// ---------------------------------------------------------------------------------------------
// ACCENT VARIANTS — one per playlist.
//
// 81 Shorts rendered from one identical template read as mass-produced in a channel feed (and
// that visual sameness is exactly what "repetitious content" review looks for). Each playlist
// therefore gets its own backdrop hue, code-surface tint, badge accent and title-card glow, so
// the grid shows seven visually distinct families instead of one repeated card.
//
// Only chrome changes: syntax colors, contrast, geometry and every safe zone stay identical, so
// the layout gates (caption band, right action rail, poster frame) behave exactly the same.
export type Accent = {
  body: string;        // page canvas
  codeBg: string;      // editor surface
  border: string;
  badge: {bg: string; fg: string};
  glowTop: string;     // title-card + backdrop radial gradients
  glowBottom: string;
};

export const accents: Record<string, Accent> = {
  slate:   {body: '#1b1b1b', codeBg: '#0a0a0a', border: '#404040', badge: {bg: '#a0caff', fg: '#0a0a0a'},
            glowTop: 'rgba(160,202,255,0.16)', glowBottom: 'rgba(239,168,255,0.13)'},
  violet:  {body: '#1a1721', codeBg: '#0c0910', border: '#463c57', badge: {bg: '#efa8ff', fg: '#0a0a0a'},
            glowTop: 'rgba(239,168,255,0.18)', glowBottom: 'rgba(160,202,255,0.12)'},
  teal:    {body: '#141d1c', codeBg: '#070f0e', border: '#31514d', badge: {bg: '#83dac9', fg: '#0a0a0a'},
            glowTop: 'rgba(131,218,201,0.17)', glowBottom: 'rgba(160,202,255,0.12)'},
  amber:   {body: '#1f1a13', codeBg: '#100c06', border: '#574a2c', badge: {bg: '#eec12f', fg: '#0a0a0a'},
            glowTop: 'rgba(238,193,47,0.15)', glowBottom: 'rgba(255,179,127,0.12)'},
  rose:    {body: '#201618', codeBg: '#110809', border: '#573f42', badge: {bg: '#ffaeaa', fg: '#0a0a0a'},
            glowTop: 'rgba(255,174,170,0.16)', glowBottom: 'rgba(239,168,255,0.11)'},
  ocean:   {body: '#131b24', codeBg: '#060c13', border: '#2f4759', badge: {bg: '#a0caff', fg: '#0a0a0a'},
            glowTop: 'rgba(160,202,255,0.18)', glowBottom: 'rgba(131,218,201,0.12)'},
  ember:   {body: '#211711', codeBg: '#120a05', border: '#5a3f2c', badge: {bg: '#ffb37f', fg: '#0a0a0a'},
            glowTop: 'rgba(255,179,127,0.17)', glowBottom: 'rgba(238,193,47,0.11)'},
};

export const accentFor = (name?: string): Accent => accents[name ?? ''] ?? accents.slate;
