from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
import asyncio

class DialogTurn(BaseModel):
    role: str  # "user" or "bot"
    content: str
    timestamp: str = datetime.now().isoformat()

class ContextConfig:
    def __init__(self, max_history: int = 5, persist: bool = False):
        self.max_history = max_history
        self.persist = persist

class ContextService:
    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        self.contexts: Dict[str, List[DialogTurn]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self.locks:
            self.locks[session_id] = asyncio.Lock()
        return self.locks[session_id]

    async def add_turn(self, session_id: str, turn: DialogTurn) -> None:
        async with await self._get_lock(session_id):
            if session_id not in self.contexts:
                self.contexts[session_id] = []
            
            self.contexts[session_id].append(turn)
            
            if len(self.contexts[session_id]) > self.config.max_history:
                self.contexts[session_id].pop(0)

    async def get_context(self, session_id: str) -> List[DialogTurn]:
        return self.contexts.get(session_id, []).copy()

    async def clear_context(self, session_id: str) -> None:
        async with await self._get_lock(session_id):
            if session_id in self.contexts:
                del self.contexts[session_id]