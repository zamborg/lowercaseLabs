-- M12 final cleanup: drop legacy public.memories table and delete old core Google vault rows.
-- Verified live data now lives in mod_memory.memories and mod_gmail/mod_gcal schemas.
-- Name-keyed tool_tiers entries remain intact.

DROP TABLE IF EXISTS public.memories;
DELETE FROM credentials WHERE issuer = 'google';
