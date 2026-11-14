# Multi-user design

A short note on how the app keeps several accounts apart. The full, current description is in
[ARCHITECTURE.md](../ARCHITECTURE.md) (section 4.2 for accounts and identity, section 11 for isolation) and
[DEPLOYMENT.md](../DEPLOYMENT.md) (section 9 for running accounts); where this note and the code differ, the
code and those documents win.

## Decisions

- Built-in accounts, created by an admin; no self-registration and no external identity provider.
- Passwords hashed with argon2id; secure session cookies; API tokens for scripts.
- Fully offline.
- Private data: admins manage accounts but cannot read anyone else's documents, chats, memories or audit
  details.
- Identity between services is an Ed25519-signed assertion with a short lifetime.

## What had to change

A single-user app keeps many things global. Each of these became per user:

1. **Document ids.** The id of an upload is salted with the owner (`sha256(owner_id:content_sha256)`), so
   identical bytes from two users are two documents.
2. **Same-name versions.** A new `report.pdf` replaces only its owner's older `report.pdf`.
3. **Files.** Originals, figure assets, table files and chunk texts live in one folder per user, and every id
   is validated before a path is built.
4. **Index reads.** Every Qdrant read filters on the owner (a tenant index on `user_id`), including reads by
   point id, so a guessed chunk id returns nothing.
5. **Chats and memories.** Every query has the owner in its WHERE clause; the memory cap, duplicates and
   eviction are per user.
6. **Audit log.** Per-user numbering and hash chains, per-user encryption keys, and crypto-shredding when an
   account is deleted.
7. **Gateway.** nginx asks `auth` on every `/api` request (`auth_request`), checks the Origin, and forwards a
   signed identity; backends refuse calls without a valid one.
8. **Browser state.** The document selection the browser remembers is kept per user, and API responses are
   not cached (`Cache-Control: no-store`).

## Components

- **`auth` service.** Accounts, sessions, API tokens, lockout and the `auth-admin` command line, in its own
  volume and container, away from every service that handles document content.
- **Identity assertions.** The gateway forwards the caller's identity as an Ed25519-signed assertion; each
  backend verifies it and scopes every read and write to that user. Background work keeps the caller's
  identity. The ingestion worker holds no signing key: it takes the owner from the registry row it claims.
- **Cascading delete.** Deleting an account removes its documents, index entries, files, chats and memories,
  and erases its audit details, service by service, with a retry for a delete that stopped half way.
- **Legacy data.** An install from before accounts is adopted by one account with `auth-admin
  migrate-legacy` (DEPLOYMENT.md, section 4.10).

## Tests

- Unit tests in each service cover another user's ids on every route and store method, owner-salted ids,
  per-owner versions and paths, the owner filter in Qdrant, and the delete cascade.
- `scripts/multiuser_smoke.py` is a live isolation test through the gateway: it creates two temporary accounts
  A and B, has B try every route on A's data (documents, chunks, assets, chat, threads, memories, search, audit
  and the admin routes), checks header spoofing, CSRF and Origin rules, cookie flags, lockout, disabling,
  password resets and concurrent use, then deletes A and checks that nothing of A is left. See its docstring
  and DEPLOYMENT.md, section 9.7.

## Residual risks

- **Host access sees everything.** Isolation is enforced by the application; anyone with Docker or root
  access on the host can read every volume and reset passwords.
- **Admin password resets.** The admin sees the temporary password; the forced change at the next sign-in,
  revoked sessions and an event in the user's own audit trail soften this.
- **Shared parser and stores.** The ingestion worker, the SQL sandbox and Qdrant handle every user's
  documents, so a parser exploit could expose documents across users (not chats, memories or audit).
- **Side channels.** Collection-wide BM25 statistics, the model server's prompt cache and GPU contention.
  There are no per-user quotas yet.
- **Revocation lag.** An answer already streaming continues after its session is revoked, for up to 15
  minutes; an identity assertion lives for at most two minutes.
- **Lockout can be abused.** Anyone who knows a username can lock it for a while by failing sign-ins.
- **Backups.** Deletion and crypto-shredding do not reach backups made earlier.
