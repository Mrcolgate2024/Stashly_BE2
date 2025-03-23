from enum import Enum

class PersonaMode(str, Enum):
    friendly = "friendly"
    rogue = "rogue"

persona_prompts = {
    "friendly": "You are a warm, supportive, funny and helpful financial analyst named Stashly. Explain things clearly and positively.",
    "rogue": "You are a blunt, sarcastic, dry and brutally honest financial assistant. Be direct, no fluff."
}

class Settings:
    def __init__(self):
        self.model = "gpt-4o"  # default
        self.temperature = 0.3
        self.persona_mode = PersonaMode.friendly
        self.memory_window = 5
        self.exposure_xml_dir = "data/funds"  # directory for XMLs
