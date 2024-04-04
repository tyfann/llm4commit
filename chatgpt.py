from openai import OpenAI
import json

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-0rLvuRkMiD4Mw25QYygh6rUlZVjpQWNGNF4yez7z3PZ7yCOm",
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.cn/v1"
)

def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=messages)
    print('msg:' + completion.choices[0].message.content)


if __name__ == '__main__':
    with open('selected_data.json', 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    for message in messages[:144]:  # 选择前面的144个消息
        try:
            prompt = [{'role': 'user','content': f"Here is a code diff: \n {message['diff']} \n Generate the commit message based on git diff(within 30 words):"},]
            gpt_35_api(prompt)
        except Exception as e:
            print(e)
            break