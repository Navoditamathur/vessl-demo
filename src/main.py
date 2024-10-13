import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def chat_with_ai(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def main():
    print("Chatbot: Hello! I'm a simple chatbot. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = chat_with_ai(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
