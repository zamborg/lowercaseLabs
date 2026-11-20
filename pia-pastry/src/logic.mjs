const FINAL_PURCHASE_RE =
  /\b(place order|pay(?: now)?|purchase|complete order|submit order)\b/iu;

export function chooseNextDropTitle(headings) {
  const normalized = headings.map((text) => text.trim());
  const start = normalized.findIndex((text) => text === "Next Drop");
  if (start < 0) return null;

  const laterHeadings = normalized.slice(start + 1);
  const stopOffset = laterHeadings.findIndex(
    (text) => text === "Frequently Asked Questions" || text === "Past Drops",
  );
  const window =
    stopOffset < 0 ? laterHeadings : laterHeadings.slice(0, stopOffset);

  return (
    window.find(
      (text) =>
        text &&
        !/hasn't scheduled|sign ?up|what's a drop/iu.test(text),
    ) ?? null
  );
}

export function pickProduct(buttonLabels, productOrder) {
  for (const product of productOrder) {
    const match = buttonLabels.find((label) =>
      label.toLocaleLowerCase().includes(product.toLocaleLowerCase()),
    );
    if (!match) continue;
    return {
      product,
      label: match,
      soldOut: /\bsold out\b/iu.test(match),
    };
  }
  return null;
}

export function preferenceRank(label, preferences) {
  const folded = label.toLocaleLowerCase();
  const rank = preferences.findIndex((preference) =>
    folded.includes(preference.toLocaleLowerCase()),
  );
  return rank < 0 ? Number.POSITIVE_INFINITY : rank;
}

export function choosePreferredLabel(labels, preferences) {
  if (preferences.length === 0) return null;
  return (
    labels
      .map((label, index) => ({
        label,
        index,
        rank: preferenceRank(label, preferences),
      }))
      .filter(({ rank }) => Number.isFinite(rank))
      .sort((a, b) => a.rank - b.rank || a.index - b.index)[0]?.label ?? null
  );
}

export function isPickupDateLabel(label) {
  const normalized = label.replace(/\s+/gu, " ").trim();
  if (!normalized || normalized.length > 80) return false;
  return /^(?:today|tomorrow|mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|\d{1,2}\/\d{1,2}(?:\/\d{2,4})?)\b/iu.test(
    normalized,
  );
}

export function isPickupTimeLabel(label) {
  const normalized = label.replace(/\s+/gu, " ").trim();
  if (!normalized || normalized.length > 80) return false;
  return /^\d{1,2}:\d{2}\s*(?:AM|PM)\b/iu.test(normalized);
}

export function chooseRandomMatchingLabel(
  labels,
  predicate,
  random = Math.random,
) {
  const candidates = labels.filter(predicate);
  if (candidates.length === 0) return null;
  const index = Math.min(
    Math.floor(random() * candidates.length),
    candidates.length - 1,
  );
  return candidates[index];
}

export function isFinalPurchaseLabel(label) {
  return FINAL_PURCHASE_RE.test(label);
}

export function extractDisplayedTotal(text) {
  const matches = [
    ...text.matchAll(
      /(?:order\s+)?total(?:\s*(?:usd))?\s*:?\s*\$\s*([0-9]+(?:\.[0-9]{1,2})?)/giu,
    ),
  ];
  if (matches.length === 0) return null;
  return Number(matches.at(-1)[1]);
}

export function parseBoolean(value, fallback = false) {
  if (value == null || value === "") return fallback;
  if (/^(1|true|yes|on)$/iu.test(value)) return true;
  if (/^(0|false|no|off)$/iu.test(value)) return false;
  throw new Error(`Expected a boolean, received "${value}".`);
}
