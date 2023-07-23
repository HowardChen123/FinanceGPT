from typing import Union
import json

class Prompter:

    def __init__(self):
        with open("utils/template.json") as fp:
            self.template = json.load(fp)

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
    ) -> str:
        res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

