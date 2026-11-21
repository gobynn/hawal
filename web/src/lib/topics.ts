// Topic names in English and Sorani, shared by the demo and the results sections.
export const TOPICS: Record<string, { en: string; ku: string; color: string }> = {
  economic: { en: "Economy", ku: "ئابووری", color: "#9a6a1e" },
  health: { en: "Health", ku: "تەندروستی", color: "#3f7a5a" },
  "science & technology": { en: "Science & Tech", ku: "زانست و تەکنەلۆژیا", color: "#2f5f9e" },
  social: { en: "Society", ku: "کۆمەڵایەتی", color: "#8a4a7a" },
  sport: { en: "Sport", ku: "وەرزش", color: "#b0482c" },
}

export const topic = (label: string) => TOPICS[label] ?? { en: label, ku: label, color: "#555" }

/** Same normalisation as training (01_prepare_data.py: clean). */
export function normalize(text: string): string {
  return text.replaceAll("ي", "ی").replaceAll("ى", "ی").replaceAll("ك", "ک").replaceAll("ة", "ە").replaceAll("ھ", "ه").replaceAll("ـ", "").replace(/\s+/g, " ").trim()
}

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`
