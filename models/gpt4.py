from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-tMbkq3K1iO5vf0FRMlrmzslGXJZwE0us3mve4QXuvpnZcumG",
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.cn/v1"
)

def gpt_4_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-4", messages=messages)
    print('msg:' + completion.choices[0].message.content)


if __name__ == '__main__':
    try:
        prompt = [{'role': 'user','content': "When talk about using LLMs(large language models) for commit message generation task(giving a code diff and generate the git commit message for it), let’s say we want to enhance the LLMs’ generation by applying RAG method(retrieval augmented generation), how can we design the system and achieve it?"},]
        gpt_4_api(prompt)
    except Exception as e:
        print(e)
