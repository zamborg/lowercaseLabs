-- M12a: retire legacy hosted 'memory' provider row so the 'memory' module owns the slug.
-- Legacy public.memories rows and tool_tiers remain intact until live cutover is verified.

DELETE FROM providers
WHERE slug = 'memory'
  AND kind = 'hosted';
