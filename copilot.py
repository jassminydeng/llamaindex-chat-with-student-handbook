from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
import textwrap

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
def chat_completion_request(client, messages, model="gpt-4",
                            **kwargs):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

class Copilot:
    def __init__(self):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embedding_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en"
        )
        self.index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model,
                                                     show_progress=True)
        self.retriever = self.index.as_retriever(
                        similarity_top_k=3
                        )
        
        self.system_prompt = """
            You are an expert on marketing research and your expertise is to use LLM to generate creative marketing research ideas.
        """

    def format_retrieved_info(self, nodes):
        """Format retrieved information with proper wrapping and spacing"""
        formatted_entries = []
        
        for i, node in enumerate(nodes, 1):
            source = node.metadata.get('file_name', 'Unknown source')
            # Clean and format the text
            text = node.text.strip().replace('\n', ' ')
            entry = f"Source {i}: {source}\n{text}"
            formatted_entries.append(entry)
        
        return "\n\n".join(formatted_entries)

    def ask(self, question, messages, openai_key=None):
        ### initialize the llm client
        self.llm_client = OpenAI(api_key=openai_key)

        ### use the retriever to get the answer
        nodes = self.retriever.retrieve(question)
        
        ### Format the retrieved information
        retrieved_info = self.format_retrieved_info(nodes)

        processed_query_prompt = """
            The user is asking a question: {question}

            The retrieved information and sources are:
            {retrieved_info}

            Please answer the question based on the retrieved information. If the question is not related to LLM creativity, 
            please tell the user and ask for a question related to LLM creativity.

            Please highlight the information with bold text and bullet points. Also include the source references [1], [2], etc. 
            when citing specific information from the sources.
        """
        
        processed_query = processed_query_prompt.format(
            question=question, 
            retrieved_info=retrieved_info
        )
        
        messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query}]
        response = chat_completion_request(self.llm_client, 
                                           messages=messages, 
                                           stream=True)
        
        return retrieved_info, response

def print_with_border(title, content):
    """Print content with a title and border"""
    width = 80
    print(f"\n{'=' * width}")
    print(f"{title}")
    print(f"{'-' * width}")
    print(content)
    print(f"{'=' * width}")

if __name__ == "__main__":
    ### get openai key from user input
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = input("Please enter your OpenAI API Key (or set it as an environment variable OPENAI_API_KEY): ")
    copilot = Copilot()
    messages = []
    
    while True:
        question = input("\nPlease ask a question: ")
        retrieved_info, answer = copilot.ask(question, messages=messages, openai_key=openai_api_key)
        
        # Display retrieved information
        print_with_border("Retrieved Information and Sources", retrieved_info)
        
        # Display answer
        print_with_border("Answer", "")
        if isinstance(answer, str):
            print(answer)
        else:
            answer_str = ""
            for chunk in answer:
                content = chunk.choices[0].delta.content
                if content:
                    answer_str += content
                    print(content, end="", flush=True)
            print()
            answer = answer_str

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    ### get openai key from user input
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = input("Please enter your OpenAI API Key (or set it as an environment variable OPENAI_API_KEY): ")
    copilot = Copilot()
    messages = []
    while True:
        question = input("Please ask a question: ")
        retrieved_info, answer = copilot.ask(question, messages=messages, openai_key=openai_api_key)
        
        print("\nRetrieved Information and Sources:")
        print("---------------------------------")
        print(retrieved_info)
        print("\nAnswer:")
        print("-------")
        
        if isinstance(answer, str):
            print(answer)
        else:
            answer_str = ""
            for chunk in answer:
                content = chunk.choices[0].delta.content
                if content:
                    answer_str += content
                    print(content, end="", flush=True)
            print()
            answer = answer_str

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})



# from openai import OpenAI
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# import os
# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
# def chat_completion_request(client, messages, model="gpt-4o",
#                             **kwargs):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             **kwargs
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# class Copilot:
#     def __init__(self):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         embedding_model = HuggingFaceEmbedding(
#             model_name="BAAI/bge-small-en"
#         )
#         self.index = VectorStoreIndex.from_documents(docs, embed_model = embedding_model,
#                                                      show_progress=True)
#         self.retriever = self.index.as_retriever(
#                         similarity_top_k=3
#                         )
        
#         self.system_prompt = """
#             You are an expert on marketing research and your expertise is to use LLM to generate creative markeing research idea.
#         """

#     def ask(self, question, messages, openai_key=None):
#         ### initialize the llm client
#         self.llm_client = OpenAI(api_key = openai_key)

#         ### use the retriever to get the answer
#         nodes = self.retriever.retrieve(question)
#         ### make answer a string with "1. <>, 2. <>, 3. <>"
#         retrieved_info = "\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])
        

#         processed_query_prompt = """
#             The user is asking a question: {question}

#             The retrived information is: {retrieved_info}

#             Please answer the question based on the retrieved information. If the question is not related to LLM creativity, 
#             please tell the user and ask for a question related to LLM creativity.

#             Please highlight the information with bold text and bullet points.
#         """
        
#         processed_query = processed_query_prompt.format(question=question, 
#                                                         retrieved_info=retrieved_info)
        
#         messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query}]
#         response = chat_completion_request(self.llm_client, 
#                                            messages = messages, 
#                                            stream=True)
        
#         return retrieved_info, response

# if __name__ == "__main__":
#     ### get openai key from user input
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         openai_api_key = input("Please enter your OpenAI API Key (or set it as an environment variable OPENAI_API_KEY): ")
#     copilot = Copilot()
#     messages = []
#     while True:
#         question = input("Please ask a question: ")
#         retrived_info, answer = copilot.ask(question, messages=messages, openai_key=openai_api_key)
#         ### answer can be a generator or a string

#         #print(retrived_info)
#         if isinstance(answer, str):
#             print(answer)
#         else:
#             answer_str = ""
#             for chunk in answer:
#                 content = chunk.choices[0].delta.content
#                 if content:
#                     answer_str += content
#                     print(content, end="", flush=True)
#             print()
#             answer = answer_str

#         messages.append({"role": "user", "content": question})
#         messages.append({"role": "assistant", "content": answer})
