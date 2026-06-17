from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


VISIBILITY_VALUES = {"PUBLIC", "PRIVATE"}
STATE_VALUES = {"ACTIVE", "DELETED"}


@dataclass
class MemoryFact:
    """A single structured personal fact.

    factActor is always the user ("self") in this single-user system, so it is
    not stored as a column. The core is the (object, content) pair plus
    lifecycle metadata.
    """

    fact_content: str
    fact_object: str = ""
    visibility: str = "PUBLIC"
    state: str = "ACTIVE"
    source: str = "chat"  # "chat" | "note"
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
