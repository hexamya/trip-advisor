from groq import Groq, NOT_GIVEN
import json


class Agent:
    def __init__(
            self,
            role: str = None,
            goal: str = None,
            backstory: str = None,
            tools: list = None,
            model: str = 'llama3-groq-70b-8192-tool-use-preview'
    ):
        self.client = Groq()
        self.model = model
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = list(map(lambda t: t.copy(), tools)) if tools else []
        self.messages = []

        self.available_functions = {}
        for tool in self.tools:
            self.available_functions[tool['function']['name']] = tool.pop('tool')

    def create_system_prompt(self):
        """Generate a system prompt based on agent's role, goal, and backstory"""
        system_prompt = f"You are an AI agent with the role of {self.role}.\n"
        if self.goal:
            system_prompt += f"Your goal is to {self.goal}.\n"
        if self.backstory:
            system_prompt += f"Here is some context about you: {self.backstory}\n"
        system_prompt += "Use the tools provided to assist the user with tasks and provide helpful responses."

        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def task(self, description: str, json_output: bool = False):
        """Run a conversation with the user, allowing the agent to use tools as needed"""
        self.create_system_prompt()

        self.messages.append({
            "role": "user",
            "content": description + ("\n\nPlease return JSON." if json_output else ""),
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=False,
            tools=self.tools,
            tool_choice="auto",
            max_tokens=4096,
            response_format={"type": "json_object"} if False else NOT_GIVEN
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.available_functions.get(function_name)
                function_args = json.loads(tool_call.function.arguments)

                if function_to_call:
                    function_response = function_to_call(**function_args)

                    self.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": function_response,
                    })

        second_response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        response = second_response.choices[0].message.content

        if json_output:
            response = json.loads(response)

        return response
