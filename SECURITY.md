# Security policy

Please report suspected vulnerabilities privately to the repository owners
before opening a public issue. Include affected versions, impact, and a minimal
reproduction when possible.

Never include API tokens, model-registry credentials, private dataset content,
internal hostnames, or infrastructure paths in an issue or pull request.

The reproduction workflow executes a pinned third-party checkout and launches
a local model server. Review the patch and generated commands before running
it on shared infrastructure. Use `--dry-run` for inspection.

