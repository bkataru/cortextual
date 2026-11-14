-- cortextual schema, version 1.
--
-- A released migration is never edited. Changes go in a new, numbered file.
--
-- Conventions:
--   * ids are 16-byte BLOBs; content hashes are 32-byte BLOBs
--   * timestmaps are INTEGER microseconds since the Unix epoch
--   * tenant_id, user_id, agent_id, session_id hold the parts of a Scope;
--     agent_id and session_id are NULL when the scope leaves them open
--   * enum-like columns hold the stable names from the Rust enums, guarded
--     by CHECK constraints
--   * seq is the insertion order, so listings never depend on id order
--     within the same millisecond


-- Episodes: the immutable log of what happened. ------------------------------

CREATE TABLE episode (
    seq          INTEGER PRIMARY KEY,
    id           BLOB    NOT NULL UNIQUE CHECK (length(id) = 16),
    tenant_id    TEXT    NOT NULL,
    user_id      TEXT    NOT NULL,
    agent_id     TEXT,
    session_id   TEXT,
    kind         TEXT    NOT NULL CHECK (kind IN (
                     'user_message', 'assistant_message', 'tool_call',
                     'tool_result', 'observation', 'document')),
    content      TEXT    NOT NULL CHECK (length(content) > 0),
    content_hash BLOB    NOT NULL CHECK (length(content_hash) = 32),
    occurred_at  INTEGER NOT NULL,
    ingested_at  INTEGER NOT NULL
);

CREATE INDEX episode_by_scope_time ON episode (tenant_id, user_id, occurred_at);
CREATE INDEX episode_by_hash ON episode (tenant_id, user_id, content_hash);

CREATE TRIGGER episode_is_immutable
BEFORE UPDATE ON episode
BEGIN
    SELECT RAISE(ABORT, 'episodes are immutable');
END;

-- Memories: knowledge derived from episodes. ---------------------------------

CREATE TABLE memory (
    seq           INTEGER PRIMARY KEY,
    id            BLOB    NOT NULL UNIQUE CHECK (length(id) = 16),
    tenant_id     TEXT    NOT NULL,
    user_id       TEXT    NOT NULL,
    agent_id      TEXT,
    session_id    TEXT,
    kind          TEXT    NOT NULL CHECK (kind IN (
                      'semantic', 'episodic', 'procedural', 'reflection')),
    text          TEXT    NOT NULL CHECK (length(text) > 0),
    importance    REAL    NOT NULL CHECK (importance BETWEEN 0 AND 10),
    confidence    REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    trust         TEXT    NOT NULL CHECK (trust IN (
                      'external', 'tool', 'user', 'system')),
    status        TEXT    NOT NULL CHECK (status IN (
                      'active', 'superseded', 'archived', 'quarantined')),
    valid_from    INTEGER,
    valid_to      INTEGER,
    recorded_at   INTEGER NOT NULL,
    superseded_at INTEGER,
    superseded_by BLOB    REFERENCES memory (id),
    last_accessed INTEGER NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),

    CHECK (valid_from IS NULL OR valid_to IS NULL OR valid_from <= valid_to),
    CHECK ((status = 'superseded') = (superseded_at IS NOT NULL)),
    CHECK ((status = 'superseded') = (superseded_by IS NOT NULL)),
    CHECK (superseded_at IS NULL OR superseded_at >= recorded_at),
    CHECK (superseded_by IS NULL OR superseded_by <> id)
);

CREATE INDEX memory_by_scope_status ON memory (tenant_id, user_id, status);
CREATE INDEX memory_by_successor ON memory (superseded_by)
    WHERE superseded_by IS NOT NULL;

-- What a memory is made of never changes. Only its bookkeeping does.
CREATE TRIGGER memory_core_is_immutable
BEFORE UPDATE OF seq, id, tenant_id, user_id, agent_id, session_id, kind, text,
                 importance, confidence, trust, valid_from, recorded_at
ON memory
BEGIN
    SELECT RAISE(ABORT, 'a memory''s identity, scope, text, scores, trust, and recording time are immutable');
END;

-- Once superseded, a memory is history: its end state is final.
CREATE TRIGGER memory_supersession_is_final
BEFORE UPDATE OF status, valid_to, superseded_at, superseded_by
ON memory
WHEN old.status = 'superseded'
BEGIN
    SELECT RAISE(ABORT, 'a superseded memory is final');
END;

-- Provenance: which episodes each memory was derived from, in order. ---------

CREATE TABLE memory_source (
    memory_id  BLOB    NOT NULL REFERENCES memory (id),
    episode_id BLOB    NOT NULL REFERENCES episode (id),
    position   INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (memory_id, episode_id)
    UNIQUE (memory_id, position)
) WITHOUT ROWID;

CREATE INDEX memory_source_by_episode ON memory_source (episode_id);

CREATE TRIGGER memory_source_is_immutable
BEFORE UPDATE ON memory_source
BEGIN
    SELECT RAISE(ABORT, 'provenance is immutable');
END;

-- Full-text search over memory text (BM25 ranking comes with the index). -----

CREATE VIRTUAL TABLE memory_fts USING fts5 (
    text,
    content = 'memory',
    content_rowid = 'seq',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER memory_fts_after_insert
AFTER INSERT ON memory
BEGIN
    INSERT INTO memory_fts (rowid, text) VALUES (new.seq, new.text);
END;

CREATE TRIGGER memory_fts_after_delete
AFTER DELETE ON memory
BEGIN
    INSERT INTO memory_fts (memory_fts, rowid, text)
    VALUES ('delete', old.seq, old.text);
END;
