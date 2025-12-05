import json
import requests
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load .env automatically
load_dotenv()

SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma:2b")


def search_web(query: str) -> str:
    if not SERPER_API_KEY:
        print("DEBUG: SERPER_API_KEY not set.")
        return "Error"

    print(f"DEBUG: Using SERPER KEY to search web")
    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, data=payload)

        print(f"DEBUG: Serper response status code: {response.status_code}")
        response.raise_for_status()
        results = response.json()

        if not results.get("organic"):
            return "No results found."

        output = "SEARCH RESULTS:\n"
        for item in results["organic"][:5]:
            output += f"title: {item.get('title', 'N/A')}\n"
            output += f"link: {item.get('link', 'N/A')}\n"
            output += f"snippet: {item.get('snippet', 'N/A')}\n\n"
        return output

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during web search: {e}")
        return "Error"


def browse_site(url: str) -> str:
    print(f"DEBUG: Browsing site: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
            'Connection': 'keep-alive'
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        if not text:
            return "No readable content found."

        print(f"DEBUG: Successfully extracted text from site: {url}")
        return text[:5000]

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while browsing the site {url}: {e}")
        return "Error"


def call_gemma_ollama(prompt: str, output_format: str = 'json') -> str:
    print(f"Thinking with gemma: {OLLAMA_MODEL}")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # Ollama returns: { "response": "text...", ... }
        return result.get("response", "").strip()

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return "Error"


def run_concierge_agent(goal: str, history: list) -> str:
    # History formatter
    formatted_history = "\n".join(history)

    # -----------------------------
    # STEP 1: GENERATE SEARCH QUERY
    # -----------------------------
    prompt1 = f"""
    You are a helpful concierge agentt. Your task is to understand a user's request and generate concise, effective search query to find the information they need. 

    Conversation history: 
    ---
    {formatted_history}
    ---

    User's latest requests: "{goal}"

    Based on the request and conversation history, generate a search query that will help find the most relevant information.
    The query should be 3-5 words. 
    Respond ONLY with the search query and nothing else.
    """

    search_query = call_gemma_ollama(prompt1, output_format='text').strip()
    search_query = search_query.replace('"', '')  # fix

    # -----------------------------
    # STEP 2: SEARCH THE WEB
    # -----------------------------
    search_results = search_web(search_query)
    print(search_results)

    # -----------------------------
    # STEP 3: PICK TOP URLS
    # -----------------------------
    prompt2 = f""" 
    You are a smart web navigator. Your task is to analuse Google search results and select the most promising URLs to find the answer to a user's goal. Abvoid generic homepages (like yelp.com or google.com) and prefer specific articles or lists or maps. 

    User's goal: "{goal}"

    Search results:

    ---
    {search_results}
    ---

    Based on the user's goal and the search results, which are the top 2-3 most promising and specific URLs to visit to find the information needed?
    Respond ONLY with a list of URLs, one per line. 
    """

    browse_urls_str = call_gemma_ollama(prompt2, output_format='text').strip()
    browse_urls = [url.strip() for url in browse_urls_str.splitlines() if url.strip().startswith("http")]

    # -----------------------------------------
    # STEP 3B: If no URLs -> summarize snippets
    # -----------------------------------------
    if not browse_urls:
        print("--- Could not identify URLs to browse. Summarizing snippets. ---")

        prompt_summarize_snippets = f""" 
        You are helpful concierge agent. Your task is to summarize the search results snippets to provide the user with useful information.

        User's goal: "{goal}"
        Search results:
        ---
        {search_results}
        ---

        Please provide a summary of the information found in the search results snippets that addresses the user's goal. Do not suggest browsing any URLs.
        """

        final_summary = call_gemma_ollama(prompt_summarize_snippets, output_format='text').strip()
        print(final_summary)
        return final_summary

    # -----------------------------
    # STEP 4: BROWSE SELECTED URLS
    # -----------------------------
    all_website_texts = []
    for url in browse_urls:
        site_text = browse_site(url)
        if not site_text.startswith("Error"):
            all_website_texts.append((url, site_text))
        else:
            print(f"--- Skipping URL due to browsing error: {url} ---")

    if not all_website_texts:
        return "Could not retrieve information from any of the selected websites."

    aggregated_texts = "\n\n".join(
        [f"URL: {url}\nContent:\n{content}" for url, content in all_website_texts]
    )

    # -----------------------------
    # STEP 5: FINAL SUMMARY
    # -----------------------------
    prompt3 = f"""
    You are meticulous and trustworthy concierge agent. Your primary goal is to provide a clear, concise, and above all, ACCURATE answer to user's request by synthesizing information from multiple web sources.
    
    User's goal: {goal}

    You have gathered the following information from various websites:
    ---
    {aggregated_texts}
    ---

    Fact check and synthesize: 
    Based on the information provide, generate a comphrensive summary that directly addresses the user's goal. Ensure that all information is accurate and verifiable from the provided sources. If there are conflicting pieces of information, highlight them and provide context.
    Before including any business or item in your summary, you MUST verify that it meets all the specific criteria outlined in the user's goal. If you cannot find explicit confirmation that a business meets a criteria, do NOT include it in your summary.

    Format your response clearly for user. It listing places, use bullet points. 
    """

    final_summary = call_gemma_ollama(prompt3, output_format='text').strip()
    print("\n================ FINAL OUTPUT ================\n")
    print(final_summary)
    print("\n==============================================\n")

    return final_summary


def main():
    if not SERPER_API_KEY:
        print("Error: SERPER_API_KEY environment variable not set.")
        return
    print("Welcome to the Concierge Agent!")
    print("I can remember the conversation history to provide better assistance.")
    print("Make your Ollama running in the background.")

    while True:
        user_goal = input("Please enter your request (or type 'exit' to quit): ")
        if user_goal.lower() == 'exit':
            print("Goodbye!")
            break
        agent_summary = run_concierge_agent(user_goal, history=[])
        history = [f"User: {user_goal}", f"Agent: {agent_summary}"]
        

if __name__ == "__main__":
    main()
