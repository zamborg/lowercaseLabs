-- Seed memory_list (read) for any existing principals initialized before M8.4

INSERT INTO tool_tiers (principal_id, tool_pattern, tier)
SELECT id, 'memory_list', 'read'
FROM principals
ON CONFLICT (principal_id, tool_pattern) DO NOTHING;
