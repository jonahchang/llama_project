import json
import yfinance as yf
import python_weather
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
print("Loading model and tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Model and tokenizer loaded successfully!", flush=True)

# Configure a Hugging Face pipeline
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Integrate Hugging Face pipeline with LangChain
llm = HuggingFacePipeline(pipeline=text_pipeline)
print("LangChain pipeline created.", flush=True)

# Define prompt template
template = """
You are an assistant that uses the proper function calls to directly provide directly quantifiable answers to user queries. Do not provide instructions on how to call the right functions. Only provide answers to user queries.

User Query: {user_query}
"""

prompt = PromptTemplate(template=template, input_variables=["user_query"])
chain = LLMChain(prompt=prompt, llm=llm)

# Define functions
async def get_current_weather(location: str) -> str:
    logger.info(f"Fetching weather for location: {location}")
    try:
        async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
            weather = await client.get(location)
            temp = weather.current.temperature
            description = weather.current.description
            result = f"The weather in {location} is {description} with a temperature of {temp}°F."
            logger.info(f"Weather result: {result}")
            return result
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return "Weather data unavailable."

def get_stock_price(ticker: str) -> str:
    logger.info(f"Fetching stock price for ticker: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'][-1]
        result = f"The stock price of {ticker} is {current_price:.2f} USD."
        logger.info(f"Stock price result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error fetching stock price: {e}")
        return "Stock price unavailable."

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

def extract_and_parse_json(response):
    import re
    # Improved regex to match valid JSON objects
    json_match = re.search(r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})', response, re.S)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            print(f"Error decoding JSON: {e}", flush=True)
    else:
        logger.error("No valid JSON found in the response.")
        print("No valid JSON found in the response.", flush=True)
    return None

def call_function(user_query, method="invoke"):
    logger.info(f"User query: {user_query}")

    try:
        if method == "invoke":
            response = chain.invoke({"user_query": user_query})
            print(f"Response received: {response}", flush=True)
        elif method == "stream":
            response = "".join([str(resp) for resp in chain.stream({"user_query": user_query})])
            print(f"Response received: {response}", flush=True)
        elif method == "batch":
            print("Batching response...", flush=True)
            responses = chain.batch([{"user_query": user_query} for _ in range(2)])
            for idx, response in enumerate(responses):
                print(f"Batch response {idx + 1}: {response}", flush=True)
                if isinstance(response, dict):
                    response = json.dumps(response)
                elif not isinstance(response, str):
                    response = str(response)

                parsed_response = extract_and_parse_json(response)
                if not parsed_response:
                    continue  # Skip to next batch response if parsing fails

                function_name = parsed_response.get("function")
                arguments = parsed_response.get("arguments", {})

                if function_name in available_functions:
                    function_to_call = available_functions[function_name]

                    # Handle async functions
                    if asyncio.iscoroutinefunction(function_to_call):
                        result = asyncio.run(function_to_call(**arguments))
                    else:
                        result = function_to_call(**arguments)
                    logger.info(f"Function result for batch {idx + 1}: {result}")
                    print(f"Function result for batch {idx + 1}: {result}", flush=True)
                else:
                    logger.error("No valid function found in the response.")
                    print("No valid function found in the response.", flush=True)

    except Exception as e:
        logger.error(f"Error parsing response or calling function: {e}")
        print(f"Exception occurred: {e}", flush=True)

# Example usage
if __name__ == "__main__":
    print("Invoke Example:", flush=True)
    call_function("What is the weather in New York?", method="invoke")

    print("\nStream Example:", flush=True)
    call_function("What is the stock price of AAPL?", method="stream")

    print("\nBatch Example:", flush=True)
    call_function("Add 15 and 7", method="batch")
