# Decisions

Only record decisions after they are verified.

## D001 — Scope
Status: ACCEPTED

This phase only addresses asynchronous live frame/video persistence using Redis and MinIO.

## D002 — Real-time safety
Status: ACCEPTED

MinIO/network/video persistence work must not block the live/GPU processing path. The live path should perform only the minimum enqueue/publish operation needed by the existing architecture.

## New decisions

Add verified decisions here after each successful step. Include:
- what was decided
- evidence/test
- files involved
