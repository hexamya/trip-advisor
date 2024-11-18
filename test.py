import json

from core.agent import Agent


# Example usage
def calculate(expression):
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": f"Invalid expression: {str(e)}"})


def weather_info(city):
    """Get weather info for a city"""
    return json.dumps({"weather": f"Weather info for {city}: sunny, 25°C"})


# Define the tools directly in the constructor
tools = [
    {
        "tool": weather_info,
        "type": "function",
        "function": {
            "name": "weather_info",
            "description": "Get weather information for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "tool": calculate,
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# Create a general agent with a role, goal, backstory, and tools
agent = Agent(
    role="Weather Assistant",
    goal="provide weather information to users",
    backstory="I am an AI assistant with knowledge of weather patterns and forecasts.",
    tools=tools  # Pass the tools directly here
)

# Example prompt for calculating
user_prompt = "What is 25 * 4 + 10?"
print(agent.task(user_prompt))

# Example prompt for weather info
user_prompt = "What's the weather in Paris?"
print(agent.task(user_prompt))
