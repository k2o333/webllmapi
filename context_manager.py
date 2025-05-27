from typing import Dict, List
from datetime import datetime
from pydantic import BaseModel

class DialogContext(BaseModel):
    history: List[Dict[str, str]] = []
    variables: Dict[str, str] = {}
    
class ContextManager:
    def __init__(self, max_history=5):
        self.contexts: Dict[str, DialogContext] = {}
        self.max_history = max_history

    async def update_context(self, session_id: str, role: str, message: str):
        if session_id not in self.contexts:
            self.contexts[session_id] = DialogContext()
        
        self.contexts[session_id].history.append({
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 保持历史记录长度
        if len(self.contexts[session_id].history) > self.max_history:
            self.contexts[session_id].history.pop(0)

    async def get_context(self, session_id: str) -> List[Dict]:
        return self.contexts.get(session_id, DialogContext()).history