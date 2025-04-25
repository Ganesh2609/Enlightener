class Streamer:

    def __init__(self, save_reasoning=True):
        self.buffer = ""
        self.inside_think = False
        self.think_buffer = ""
        self.save_reasoning = save_reasoning

    def stream_to_buffer(self, token, stream=False):
        if "<think>" in token:
            self.inside_think = True
            token = token.replace("<think>", "")
        
        if "</think>" in token:
            token_parts = token.split("</think>")
            if self.inside_think:
                self.think_buffer += token_parts[0]
                if self.save_reasoning:
                    self.buffer += "<think>" + self.think_buffer + "</think>"
                self.think_buffer = ""
            self.inside_think = False
            token = token_parts[1] if len(token_parts) > 1 else ""

        if self.inside_think:
            self.think_buffer += token
        else:
            self.buffer += token
            if stream:
                print(token, end="", flush=True)

    def return_buffer(self):
        return self.buffer.strip()

    def clear_buffer(self):
        self.buffer = ""
        self.think_buffer = ""
        self.inside_think = False
