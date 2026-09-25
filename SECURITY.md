# Security policy

## Report a vulnerability privately

Email **info@epistemic.ai** with the subject **Diffstory security**. Include the affected version, impact, and a small synthetic reproduction. Do not post tokens, private source, personal data, or exploit details in a public issue. No response-time guarantee is implied.

## Threat model

Diffstory reads committed Git objects, GitHub responses, snapshots, annotations, and provider responses as untrusted data. It does not execute target-repository source or its tests. Local Git calls use argument arrays, disable external diff/text-conversion drivers, and set a no-hooks path. Network access is limited to explicit GitHub ingestion and explicitly confirmed OpenAI narration; the browser reader itself is offline.

The renderer validates structural fields, escapes code and prose, and escapes script terminators in embedded JSON. Its Content Security Policy disables network connections, external scripts, plugins, forms, and framing dependencies. The inline JavaScript is the trusted reader shipped with Diffstory, not input repository code. These controls do not make an HTML report supplied by an unrelated third party safe to open; regenerate reports with a trusted copy of the compiler.

Input limits and conservative matching reduce accidental failures; this is not a hardened sandbox or a complete defense against every resource-exhaustion input. Only analyze repositories and snapshots within your environment's approved size and trust boundaries.

## Private repositories

A standalone HTML report includes source, even when a diff is collapsed. Snapshot, report, annotation, evidence, and review-note files may contain confidential information. Keep them within the repository's access boundary. Do not upload them to public hosting or model providers without authorization.

Model narration is opt-in. Before each run, Diffstory shows the provider, destination, revisions, changed-file scope, and conservative request budget, then asks for confirmation. `--yes` bypasses that prompt for automation and should be used only where source transfer is already approved. The first adapter sends bounded source evidence and summaries to the OpenAI Responses API. It sets `store: false`; this disables Responses application-state storage for the request, but does not disable abuse-monitoring controls. OpenAI's current [API data controls](https://developers.openai.com/api/docs/guides/your-data) describe up to 30 days of abuse-monitoring retention for Responses requests by default, with organization-level controls and exceptions. Do not treat `store: false` as a zero-retention guarantee.

GitHub and model credentials are read from named environment variables and are not written into reports. Do not embed tokens in PR URLs or command arguments. Both API clients reject HTTP redirects rather than forwarding authorization to another location. Provider error messages are sanitized before display.

## Release checks

The example code and data were written for Diffstory; they were not copied from a private repository.

Before a release, an automatic check looks for names from known private samples, common access-token or private-key text, extra files, and images or other file types that are not approved. It can miss sensitive information and cannot tell whether we have permission to publish every file, so a person still needs to review the code, data, and screenshots.

GitHub runs the project's tests automatically. Those jobs can read project files, but the workflow does not pass saved API keys or passwords to its pull-request tests.

## Supported versions

Security fixes target the latest 0.3.x release. Earlier prototypes are not maintained as separate release lines. Review releases before deploying this alpha software in a sensitive workflow.
