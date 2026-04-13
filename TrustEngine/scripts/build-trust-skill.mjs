import { getConfig } from "../src/lib/config.mjs";
import { rebuildTrustArtifacts } from "../src/lib/trust-builder.mjs";

const config = getConfig();
const result = await rebuildTrustArtifacts(config);

console.log(
  JSON.stringify(
    {
      ok: true,
      generatedAt: result.compiledPolicy.generatedAt,
      suggestedRules: result.compiledPolicy.suggestedRules.length,
      skillPath: config.skillPath,
      policyPath: config.policyPath
    },
    null,
    2
  )
);
