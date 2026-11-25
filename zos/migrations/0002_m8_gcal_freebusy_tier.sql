-- Seed gcal_freebusy (read) for any existing principals initialized before M8.3

INSERT INTO tool_tiers (principal_id, tool_pattern, tier)
SELECT id, 'gcal_freebusy', 'read'
FROM principals
ON CONFLICT (principal_id, tool_pattern) DO NOTHING;
