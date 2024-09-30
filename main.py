import re
import json
import requests
import yfinance as yf
import python_weather
import asyncio
import torch
from airllm import AutoModel

# Add logging to monitor progress
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize the AirLLM model (Using a Llama-based model)
logger.info("Starting to load the Llama model...")

try:
    # Load the model
    model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Asynchronous function to fetch weather data using python-weather
async def get_current_weather(location: str) -> str:
    logger.info(f"Fetching weather for location: {location}")
    
    try:
        async with python_weather.Client() as client:
            weather = await client.get(location)
            temp = weather.temperature
            description = weather.description
            logger.info(f"Weather data fetched successfully for {location}")
            return f"{temp}°F and {description}"

    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return "Weather is unknown"

# Function to get real-time stock prices using yfinance
def get_stock_price(ticker: str) -> str:
    logger.info(f"Fetching stock price for ticker: {ticker}")
    
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'][0]
        logger.info(f"Stock price fetched successfully for {ticker}")
        return f"The current price of {ticker} is {current_price:.2f} USD"
    except Exception as e:
        logger.error(f"Error fetching stock price: {e}")
        return "Stock price is unknown"

# Basic arithmetic operations
def perform_calculation(a: int, b: int, operation: str) -> int:
    logger.info(f"Performing calculation: {a} {operation} {b}")
    
    try:
        if operation == 'add' or operation == 'plus':
            return a + b
        elif operation == 'subtract' or operation == 'minus':
            return a - b
        elif operation == 'multiply' or operation == 'times':
            return a * b
        elif operation == 'divide':
            return a // b if b != 0 else 'undefined'
        else:
            return 'Invalid operation'
    except Exception as e:
        logger.error(f"Error performing calculation: {e}")
        return 'Error in calculation'

# Functions that can be called by Llama 3.1
available_functions = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price,
    "perform_calculation": perform_calculation,
}

# Parse Llama's function call response
def parse_tool_response(response: str):
    logger.info("Parsing tool response...")
    
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)

    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            logger.info(f"Function '{function_name}' called with arguments: {args}")
            return {
                "function": function_name,
                "arguments": args,
            }
        except json.JSONDecodeError as error:
            logger.error(f"Error parsing function arguments: {error}")
            return None
    logger.warning("No function call found in response")
    return None

# Pre-process user input to detect appropriate function to call
def pre_process_input(user_input: str):
    logger.info(f"Pre-processing input: {user_input}")
    
    # Check for weather query
    if any(word in user_input.lower() for word in ["weather", "temperature", "forecast"]):
        location_match = re.search(r"in (.+)", user_input, re.IGNORECASE)
        if location_match:
            location = location_match.group(1)
            return {"function": "get_current_weather", "arguments": {"location": location}}

    # Check for stock price query
    if "stock price" in user_input.lower():
        ticker_match = re.search(r"of (.+)", user_input, re.IGNORECASE)
        if ticker_match:
            ticker = ticker_match.group(1).upper()
            return {"function": "get_stock_price", "arguments": {"ticker": ticker}}

    # Check for basic calculations
    calculation_match = re.search(r"(\d+) (plus|minus|times|add|subtract|multiply|divide) (\d+)", user_input, re.IGNORECASE)
    if calculation_match:
        a = int(calculation_match.group(1))
        operation = calculation_match.group(2).lower()
        b = int(calculation_match.group(3))
        return {"function": "perform_calculation", "arguments": {"a": a, "b": b, "operation": operation}}

    return None

# Interact with Llama model
def interact_with_llama(user_input: str):
    logger.info(f"User input received: {user_input}")
    
    toolPrompt = """
    You are an assistant capable of calling specific functions to provide information.
    You have access to the following functions:

    Use 'get_current_weather' to get the current weather in a location:
    Example: <function=get_current_weather>{"location": "New York"}</function>

    Use 'get_stock_price' to get the stock price of a company:
    Example: <function=get_stock_price>{"ticker": "AAPL"}</function>

    Use 'perform_calculation' to perform arithmetic operations:
    Example: <function=perform_calculation>{"a": 10, "b": 20, "operation": "add"}</function>

    Always respond using one of the available functions in the format <function=function_name>{"argument_name": "argument_value"}</function>.
    """

    # Pre-process the input to detect specific keywords and patterns
    parsed_response = pre_process_input(user_input)
    if parsed_response:
        function_name = parsed_response["function"]
        function_to_call = available_functions[function_name]
        args = parsed_response["arguments"]

        if asyncio.iscoroutinefunction(function_to_call):
            result = asyncio.run(function_to_call(**args))
        else:
            result = function_to_call(**args)
        
        logger.info(f"Function result: {result}")
    else:
        # Default behavior using Llama model
        logger.info("Preparing input for model...")
        messages = toolPrompt + user_input  # Combine tool prompt and user input into a single string

        logger.info("Configuring tokenizer padding token...")
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        logger.info("Tokenizing inputs...")
        try:
            input_tokens = model.tokenizer(
                messages, 
                return_tensors="pt", 
                padding=True,        
                truncation=True,     
                max_length=128       
            )

            input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

            logger.info(f"Tokenized input_ids: {input_tokens['input_ids']}")
            logger.info("Generating response from model...")

            response = model.generate(
                input_tokens['input_ids'], 
                max_new_tokens=50, 
                use_cache=True, 
                return_dict_in_generate=True
            )

            logger.info("Decoding model response...")
            response_message = model.tokenizer.decode(response.sequences[0])
            logger.info(f"Model response: {response_message}")

            parsed_response = parse_tool_response(response_message)

            if parsed_response:
                function_to_call = available_functions[parsed_response["function"]]
                if asyncio.iscoroutinefunction(function_to_call):
                    result = asyncio.run(function_to_call(**parsed_response["arguments"]))
                else:
                    result = function_to_call(**parsed_response["arguments"])
                logger.info(f"Function result: {result}")
            else:
                logger.info("No valid function found in model response")
        except Exception as e:
            logger.error(f"Error during interaction with model: {e}")
            raise

# Main loop to accept user input from terminal
if __name__ == "__main__":
    while True:
        user_input = input("Please enter your request (or type 'exit' to quit): ")
        
        if user_input.lower() == "exit":
            break
        
        interact_with_llama(user_input)
