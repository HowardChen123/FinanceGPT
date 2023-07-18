from typing import Union

class Prompter(object):

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction:
                    {instruction}
                    ### Input:
                    {input}
                    ### Response:
                    {label}
                """

    def get_response(self, output: str) -> str:
        return 