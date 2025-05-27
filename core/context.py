from typing import Dict, List

class ContextManager:
    def __init__(self):
        self.sessions: Dict[str, List[dict]] = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})
        
    def get_context(self, session_id: str, max_history=5) -> List[dict]:
        history = self.sessions.get(session_id, [])
        return history[-max_history:] if max_history > 0 else history.copy()