// Preloaded (-r) before Wrangler's CLI. yargs treats an "electron" runtime as
// having a different argv layout, so hide that marker.
try {
  Object.defineProperty(process.versions, "electron", { value: undefined });
} catch {}
