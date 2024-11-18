import uuid
from textwrap import dedent

import pymongo
import yaml
from contextlib import contextmanager
from sshtunnel import SSHTunnelForwarder
from core.agent import Agent
from core.tools import google_search_api, map_search_api


class Config:
    def __init__(self, file_path="config.yaml"):
        self.file_path = file_path
        self.config_data = self._load_config()

    def _load_config(self):
        """Loads the configuration file and returns it as a dictionary."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            print(f"Config file '{self.file_path}' not found. Starting with an empty config.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return {}

    def __getitem__(self, key, default=None):
        """Gets a configuration value using dot notation for nested keys."""
        keys = key.split('.')
        value = self.config_data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class MongoDBHandler:
    def __init__(self, config):
        config = Config(config)
        self.mongodb_config = config['mongodb']
        self.ssh_config = config['ssh']
        self.use_ssh = config['mongodb.use_ssh']
        self.connection_string = None
        self.ssh_tunnel = None
        self.client = None

    @contextmanager
    def connect(self):
        ssh_config = self.ssh_config
        mongodb_config = self.mongodb_config

        if self.use_ssh and self.ssh_tunnel:
            self.ssh_tunnel = SSHTunnelForwarder(
                ssh_address_or_host=(ssh_config['host'], ssh_config['port']),
                ssh_username=ssh_config['username'],
                ssh_password=ssh_config['password'],
                remote_bind_address=(mongodb_config['host'], mongodb_config['port'])
            )
            host = "127.0.0.1"
            port = self.ssh_tunnel.local_bind_port
        else:
            host = mongodb_config['host']
            port = mongodb_config['port']

        try:
            if self.use_ssh:
                self.ssh_tunnel.start()
            self.client = pymongo.MongoClient(
                host=host,
                port=port,
                username=mongodb_config['username'],
                password=mongodb_config['password'],
            )
            self.connection_string = f"mongodb://{mongodb_config['username']}:{mongodb_config['password']}@{host}:{port}/"
            yield self.client[mongodb_config['database']]
        finally:
            if self.use_ssh:
                self.ssh_tunnel.stop()
            self.client.close()


class Session:
    def __init__(self, session_id: int = None):
        self.session_id = session_id if session_id is not None else str(uuid.uuid4())
        self.db = MongoDBHandler("config.yaml")
        with self.db.connect() as db:
            if session_id is not None:
                self.session = next(db.sessions.find({"session_id": self.session_id}))
            else:
                self.session = {"session_id": self.session_id, "context": [], "options": [], "plan": None}
                db.sessions.insert_one(self.session)

    def update(self):
        with self.db.connect() as db:
            self.session = db.sessions.update_one({"session_id": self.session_id})

    def __getitem__(self, item):
        self.session.get(item)


class Advisor:
    def __init__(self, session_id: int):
        self.session = Session(session_id)

        self.tools = {
            "map_search_api": {
                "tool": map_search_api,
                "type": "function",
                "function": {
                    "name": "map_search_api",
                    "description": "Search for places based on a term, latitude, and longitude using the map search API. Useful for finding nearby restaurants, cafes, and other locations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "term": {
                                "type": "string",
                                "description": "The search term to look for, such as 'restaurant', 'cafe', or 'park'."
                            },
                            "lat": {
                                "type": "string",
                                "description": "The latitude coordinate of the location where the search is performed."
                            },
                            "lng": {
                                "type": "string",
                                "description": "The longitude coordinate of the location where the search is performed."
                            }
                        },
                        "required": ["term", "lat", "lng"]
                    }
                }
            },
            "google_search_api": {
                "tool": google_search_api,
                "type": "function",
                "function": {
                    "name": "google_search_api",
                    "description": "Perform a Google search and retrieve the top results with optional content inclusion.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string, such as 'Tehran concerts tickets' or 'best travel destinations'."
                            },
                            "include_content": {
                                "type": "boolean",
                                "description": "Flag to include the content snippet of the search result. Defaults to true."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        }

        self.context_analyzer_agent = Agent(
            role="Travel Context Analyzer",
            goal="Convert user travel queries into structured context summaries and identify critical missing information",
            backstory=dedent("""
            You're the first-line analyzer in a travel planning system. 
            You transform unstructured travel conversations into clear context profiles 
            that other agents use to make travel recommendations. 
            Your context summaries help the recommendation agents understand user needs 
            and make informed suggestions.
            """)
        )
        self.recommender_agent = Agent(
            role="Travel Recommendation Specialist",
            goal="Generate personalized, multi-option travel recommendations based on user context and preferences",
            backstory=dedent("""
            You're the recommendation expert in the travel planning system.
            Using context profiles from the Context Analyzer, you create structured sets of
            travel options that match user preferences. You provide multiple choices for
            destinations, dates, activities, and logistics, allowing users to mix and match
            their perfect trip components.
            """),
            tools=[self.tools["map_search_api"], self.tools["google_search_api"]]
        )
        self.planner_agent = Agent(
            role="Travel Itinerary Planner",
            goal="Create comprehensive, day-by-day travel plans by organizing all selected options into a structured, detailed itinerary",
            backstory=dedent("""
            You're the final organizer in the travel planning system.
            You take all selected destinations, accommodations, activities,
            transportation, and dining choices and transform them into a
            cohesive daily plan with all necessary details and logistics.
            """),
            tools=[self.tools["google_search_api"]]
        )
        self.context = self.session["context"]
        self.options = self.session["options"]

    def context_analyze(self, answer: str, question: str) -> dict:

        output = self.context_analyzer_agent.task(
            description=dedent(f"""
            TASK:
            Convert user queries into rich context summaries and intelligent follow-up questions.
            
            INPUT:
            {{
                "agent_question": {question},
                "user_answer: {answer},
                "previous_context": {self.context},
            }}
            
            OUTPUT:
            {{
                "user_context": "Rich descriptive text about user preferences and profile",
                "following_question": "Natural follow-up question for missing information"
            }}
            
            PROCESS:
            1. Preference Analysis
            - Explicit Preferences:
            * Directly stated destinations
            * Budget mentions
            * Timeline requirements
            * Activity requests
            
            - Implicit Preferences:
            * Travel style hints
            * Comfort level indicators
            * Cultural interest signals
            * Risk tolerance cues
        
            - Personal Context:
            * Group composition
            * Age indicators
            * Physical capabilities
            * Cultural background
            * Travel experience level        
            
            2. Context Synthesis
            - Combine all identified preferences
            - Map relationships between preferences
            - Factor in seasonal considerations
            - Consider destination-specific context
            - Analyze budget implications
            - Evaluate logistical requirements
            
            3. Follow-up Question Generation
            - Identify critical information gaps
            - Prioritize questions by impact
            - Use warm, conversational tone
            - Include relevant examples
            - Consider destination-specific factors
            
            Response Guidelines
            
            Always acknowledge and build upon previous context
            Use travel industry expertise to make informed assumptions
            Balance between gathering information and maintaining conversation flow
            Adapt tone based on user's communication style
            Provide subtle education about destinations when relevant        
            
            EXAMPLES:
            Query: "Looking for a beach vacation in Asia this December"
            Output:
            {{
                "user_context": "The traveler is seeking a warm beach destination in Asia during December's winter season. They have an international travel preference with timing that aligns with peak season in many Asian beach destinations. The interest in beaches suggests a desire for relaxation and possibly water activities.",
                "following_question": "What's your ideal trip length and are you interested more in secluded beaches like those in Thailand's islands, or lively beaches with nearby city attractions like Bali?"
            }}
        """),
            json_output=True
        )

        self.context = output["user_context"]

        return output

    def recommendation(self) -> dict:
        output = self.recommender_agent.task(
            description=dedent(f"""
            TASK:
            Follow this specific order when making recommendations:
            
            1. Destination Options
            2. Travel Dates                
            3. Accommodation Options
            4. Transportation Options
            
            INPUT:
            User Context: {self.context}
            
            Operating Rules
            
            1. Sequence Following
            - Always follow the recommendation order based on the specified context for specify more details(Destination → Dates → Accommodation → Transportation → Activities -> Dining)
            - Wait for user selection before proceeding to next step
            
            2. Recommendation Generation
            - Always provide 5-7 options for each category
            - Include at least one premium and one budget option
            - Sort by match score descending
            - Explain why each option matches user preferences
            
            3. Context Consideration
            - Reference previous selections when making new recommendations
            - Consider group composition for all suggestions
            - Factor in stated budget constraints
            - Account for seasonal factors
            
            Output Example:
            {{
                "message": "Great choice! For Bali in December, here are the best dates considering weather and your preferences:",
                "date_options": [...]
            }}
        """),
            json_output=True
        )

        return output

    def planning(self) -> dict:
        output = self.planner_agent.task(
            description=dedent(f"""
            You are a Travel Itinerary Planner that creates comprehensive, day-by-day travel plans by organizing all selected options into a structured, detailed itinerary. You take all selected destinations, accommodations, activities, transportation, and dining choices and transform them into a cohesive daily plan with all necessary details and logistics.
            
            INPUT STRUCTURE:
            The input will be a JSON object containing:
            {{
                "selected_options": {self.options},
                "user_context": "{self.context}"
            }}
            
            REQUIRED OUTPUT STRUCTURE:
            You must provide a JSON response in this exact format:
            {{
                "final_plan": {{
                    "plan_summary": {{
                        "title": "Descriptive trip title",
                        "traveler_info": {{
                            "group_size": "Number and type of travelers",
                            "travel_dates": "Full date range",
                            "budget_category": "Budget level"
                        }},
                        "overview": "Brief trip description",
                        "highlights": ["Key experiences", "Special moments", "Unique opportunities"],
                        "total_budget": {{
                            "amount": "Total in USD",
                            "breakdown": {{
                                "accommodation": "Amount",
                                "activities": "Amount",
                                "transportation": "Amount",
                                "food": "Amount",
                                "miscellaneous": "Amount"
                            }}
                        }}
                    }},
                    "essential_information": {{
                        "weather_forecast": "General weather info",
                        "required_documents": ["Necessary documents", "Visas", "Passes"],
                        "important_notes": ["Practical tips", "Local customs", "Essential apps"],
                        "health_safety": ["Local emergency numbers", "List of facilities", "Location-specific advice"]
                        "packing_recommendations": ["Weather-appropriate items", "Activity-specific gear"]
                    }},
                    "daily_itinerary": [
                        {{
                            "day": "Day number",
                            "date": "Specific date",
                            "weather_forecast": "Range in C/F Expected weather",
                            "activities": [
                                {{
                                    "time": "Start time",
                                    "activity": "Name of activity",
                                    "duration": "Length in minutes/hours",
                                    "location": {{
                                        "name": "Place name",
                                        "address": "Full address",
                                        "coordinates": "GPS coordinates",
                                        "map_link": "URL"
                                    }},
                                    "transport": {{
                                        "mode": "Type of transport",
                                        "duration": "Travel time",
                                        "cost": "Amount",
                                        "notes": "Special instructions"
                                    }},
                                    "booking_reference": "If applicable",
                                    "budget": {{
                                        "amount": "Cost in local currency and USD",
                                        "included_items": ["What's covered"],
                                        "additional_costs": ["Optional extras"]
                                    }},
                                    "tips": ["Relevant advice", "Best photo spots", "What to bring"]
                                }}
                            ],
                            "daily_notes": ["Day-specific tips", "Timing considerations", "Backup plans"]
                        }}
                    ],
                    "contingency_plans": {{
                        "weather_alternatives": ["Indoor options"],
                        "backup_activities": ["Secondary choices"],
                        "flexible_timing": ["Adjustable components"]
                    }}
                }}
            }}
            
            PLANNING RULES:
            
            1. Organization:
            - Create logical day-by-day flow
            - Balance activity levels
            - Account for travel times between locations
            - Include meal times and rest periods
            - Ensure activities are properly spaced
            
            2. Time Management:
            - Add 30-minute buffers between activities
            - Consider check-in/check-out times
            - Check venue operating hours
            - Plan around sunrise/sunset times
            - Account for peak tourist times
            
            3. Budget Tracking:
            - Include all costs in local currency and USD
            - Track running total against overall budget
            - Note included vs additional costs
            - Add booking references where applicable
            
            4. Practical Details:
            - Check weather impact on activities
            - Include all booking requirements
            - Add transportation logistics
            - Provide local tips and cultural notes
            - Include emergency information
            
            5. Documentation:
            - Start with clear summary
            - List essential information first
            - Create detailed daily schedules
            - Include backup plans
            - Add relevant maps and coordinates
            
            Remember to maintain a balance between activities and rest, consider local customs and timing, and ensure all practical details are included for each activity.
            """),
            json_output=True
        )

        return output

    def chat(self, answer: str, question: str, options: list = None):
        if options is not None:
            self.options += options
        output = self.context_analyze(answer, question)
        output["recommendation"] = self.recommendation()
        try:
            output["planning"] = self.planning()
        except:
            pass

        return output


session = Session()
advisor = Advisor(session.session_id)

output = {
    "following_question": "Hello"
}
while True:
    output = advisor.chat(input(output["following_question"] + "? "), output["following_question"])
    print(output)

