-- M12c: retire legacy core 'gcal' and 'gmail' provider rows so the standalone modules own those slugs.
-- Legacy Google credential rows in `credentials` remain intact until live modules are connected and verified.

DELETE FROM providers
WHERE slug IN ('gcal', 'gmail')
  AND NOT COALESCE((config->>'is_module')::boolean, false);
