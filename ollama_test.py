import requests
import json 

def query_ollama(prompt):
    url = "http://127.0.0.1:11434/api/generate"
    data = {
        "model": "gemma:2B",
        "prompt": prompt
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                json_line = json.loads(decoded_line)
                full_response += json_line.get('response: ', '')
                print(decoded_line)
        return full_response
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    

if __name__ == "__main__":
    prompt = "Write a short poem about the sea."
    response = query_ollama(prompt)
    if response:
        print("Full Response:")
        print(response)