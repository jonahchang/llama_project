import json
import yfinance as yf
import python_weather
import asyncio
from langchain import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(device)
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# LangChain LLM integration
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Define prompt template for LangChain function calling
template = """
You are an assistant capable of calling specific functions. Use the following functions when appropriate:

get_current_weather(location: str) -> str
- Fetches current weather information for a given location.

get_stock_price(ticker: str) -> str
- Retrieves the current stock price for a given ticker symbol.

perform_calculation(a: int, b: int, operation: str) -> int
- Performs basic arithmetic calculations.

Respond with the function and arguments in JSON format.

User Query: {user_query}
"""

# Define the LangChain PromptTemplate
prompt = PromptTemplate(template=template, input_variables=["user_query"])

# LangChain LLMChain for function calling
chain = LLMChain(prompt=prompt, llm=llm)

# Define functions
async def get_current_weather(location: str) -> str:
    logger.info(f"Fetching weather for location: {location}")
    try:
        async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
            weather = await client.get(location)
            temp = weather.temperature
            description = weather.description
            return f"{temp}°F and {description}"
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return "Weather is unknown"

def get_stock_price(ticker: str) -> str:
    logger.info(f"Fetching stock price for ticker: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'][0]
        return f"The current price of {ticker} is {current_price:.2f} USD"
    except Exception as e:
        logger.error(f"Error fetching stock price: {e}")
        return "Stock price is unknown"

def perform_calculation(a: int, b: int, operation: str) -> int:
    logger.info(f"Performing calculation: {a} {operation} {b}")
    try:
        if operation == 'add':
            return a + b
        elif operation == 'subtract':
            return a - b
        elif operation == 'multiply':
            return a * b
        elif operation == 'divide' and b != 0:
            return a // b
        return 'Invalid operation'
    except Exception as e:
        logger.error(f"Error performing calculation: {e}")
        return 'Error in calculation'

available_functions = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price,
    "perform_calculation": perform_calculation,
}

# LangChain function caller with examples of invoke, stream, and batch
def call_function(user_query, method="invoke"):
    logger.info(f"User query: {user_query}")
    
    if method == "invoke":
        response = chain.invoke({"user_query": user_query})
    elif method == "stream":
        response = chain.stream({"user_query": user_query})
    elif method == "batch":
        responses = chain.batch([{"user_query": user_query} for _ in range(3)])
        response = responses[0]  # Take the first batch result as an example
    else:
        raise ValueError("Invalid method. Choose from 'invoke', 'stream', or 'batch'.")

    try:
        parsed_response = json.loads(response)
        function_name = parsed_response.get("function")
        arguments = parsed_response.get("arguments", {})
        
        if function_name in available_functions:
            function_to_call = available_functions[function_name]
            if asyncio.iscoroutinefunction(function_to_call):
                result = asyncio.run(function_to_call(**arguments))
            else:
                result = function_to_call(**arguments)
            logger.info(f"Function result: {result}")
        else:
            logger.error("No valid function found in the response.")
    except Exception as e:
        logger.error(f"Error parsing response or calling function: {e}")

# Example usage
if __name__ == "__main__":
    print("Invoke Example:")
    call_function("What is the weather in New York?", method="invoke")

    print("\nStream Example:")
    call_function("What is the stock price of AAPL?", method="stream")

    print("\nBatch Example:")
    call_function("Add 15 and 7", method="batch")
