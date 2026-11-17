# Raw experiment attachments

## A100 validation

- Archive filename: `a100-validation-2026-11-16.tar.gz`
- Size: 8,184,488 bytes (about 7.8 MiB)
- SHA-256: `e7fae86368ea502e76e8fd920795b2bdf2c0d404d04ee95b1948d6a3ab07fe59`
- [Member manifest](a100-validation-2026-11-16.manifest.json): 85 files with individual
  sizes and SHA-256 values.
- Status: **prepared and verified locally, not uploaded**. No release URL is available.

The archive is ignored by Git and is not included in a clone. It contains the
`cloud-test-2026-11-16/` directory with server/vLLM/router logs, complete training
rollout summaries, logtree JSON, HTML reports, timing spans, configurations,
metrics, checkpoint references and historical deployment-script snapshots.
The separately committed evaluation records are under the experiment directories.

File contents are preserved byte for byte. Tar owner names are empty, timestamps
are zero and file modes are 0644. After extraction, invoke historical scripts with
`bash` or `python` explicitly; the snapshot is evidence, not the current deployment
entry point.

Excluded from the archive:

- All `wandb/` internals, including personal email/host metadata, sync databases and
  duplicated debug/output logs. These remain in the original local run directories;
  remote synchronization has not been verified.
- One duplicate pre-resume metrics backup. The retained copy and shared checksum
  are listed in the manifest.
- No secrets file, model weights, datasets, virtual environments, `local_docs/` or
  `docs/` was added to this attachment.

No historical cloud source files were deleted other than the confirmed duplicate
metrics backup. Regenerable project bytecode/lint/test caches and `.DS_Store` were
also cleared during organization. Existing project scripts were retained.

When this attachment is published to a GitHub Release, update `download_url` and
`publication_status` in the manifest and add the verified download link here. Until
then, do not describe all raw material as publicly downloadable.
