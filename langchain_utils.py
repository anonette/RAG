from langchain import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory, CombinedMemory
from langchain import PromptTemplate
from search_indexing import search_faiss_index


class SnippetsBufferWindowMemory(ConversationBufferWindowMemory):
    """
    MemoryBuffer used to hold the document snippets. Inherits from ConversationBufferWindowMemory, and overwrites the
    load_memory_variables method
    """

    index: FAISS = None
    pages: list = []
    memory_key = 'snippets'
    snippets: list = []

    def __init__(self, *args, **kwargs):
        ConversationBufferWindowMemory.__init__(self, *args, **kwargs)
        self.index = kwargs['index']

    def load_memory_variables(self, inputs) -> dict:
        """
        Based on the user inputs, search the index and add the similar snippets to memory (but only if they aren't in the
        memory already)
        """

        # Search snippets
        similar_snippets = search_faiss_index(self.index, inputs['user_messages_history'])
        # In order to respect the buffer size and make its pruning work, need to reverse the list, and then un-reverse it later
        # This way, the most relevant snippets are kept at the start of the list
        self.snippets = [snippet for snippet in reversed(self.snippets)]
        self.pages = [page for page in reversed(self.pages)]

        for snippet in similar_snippets:
            page_number = snippet.metadata['page']
            # Load into memory only new snippets
            snippet_to_add = f""
            if snippet.metadata['title'] == snippet.metadata['source']:
                snippet_to_add += f"{snippet.metadata['source']}\n"
            else:
                snippet_to_add += f"[{snippet.metadata['title']}]({snippet.metadata['source']})\n"

            snippet_to_add += f"<START_SNIPPET_PAGE_{page_number + 1}>\n"
            snippet_to_add += f"{snippet.page_content}\n"
            snippet_to_add += f"<END_SNIPPET_PAGE_{page_number + 1}>\n"
            if snippet_to_add not in self.snippets:
                self.pages.append(page_number)
                self.snippets.append(snippet_to_add)

        # Reverse list of snippets and pages, in order to keep the most relevant at the top
        # Also prune the list to keep the buffer within the define size (k)
        self.snippets = [snippet for snippet in reversed(self.snippets)][:self.k]
        self.pages = [page for page in reversed(self.pages)][:self.k]
        to_return = ''.join(self.snippets)

        return {'snippets': to_return}


def construct_conversation(prompt: str, llm, memory) -> ConversationChain:
    """
    Construct a ConversationChain object
    """

    prompt = PromptTemplate.from_template(
        template=prompt,
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False,
        prompt=prompt
    )

    return conversation


def initialize_chat_conversation(index: FAISS,
                                 model_to_use: str = 'gpt-4-1106-preview',
                                 max_tokens: int = 3500) -> ConversationChain:

    prompt_header = """You are a discourse and thematic analysis specialist who knows how to label and code snippets of texts to support research on public participation in policy. 
    Read through the collected data thoroughly to understand the depth and breadth of the content. This involves making initial observations and possibly noting down interesting aspects for further analysis. 
    The following snippets can be used to help you answer the questions:    
    {snippets}    
    The following is an example of a request and how to respond. Systematically code the data by highlighting and labeling the most notable features of the data that are relevant to the research question. Codes can be descriptive, inferential, or thematic.Example of Coding Snippets:
Snippet: We demand more transparency in the decision-making process.
Code: Demand for Transparency
Snippet: The online forum for policy feedback is user-unfriendly and inaccessible.
Code: Accessibility Issues. Answer based on the provided snippets and the conversation history. Make sure to take the previous messages in consideration, as they contain additional context.
    If the provided snippets don't include the answer, please say so, and don't try to make up an answer instead. Include in your reply the title of the document and the page from where your answer is coming from, if applicable.

    {history}    
    Customer: {input}
    """

    llm = ChatOpenAI(model_name=model_to_use, max_tokens=max_tokens)
    conv_memory = ConversationBufferWindowMemory(k=3, input_key="input")
    
    # Number of snippets that will be added to the prompt. Too many snippets and you risk both the prompt going over the
    # token limit, and the model not being able to find the correct answer
    prompt_number_snippets = 3
    snippets_memory = SnippetsBufferWindowMemory(k=prompt_number_snippets, index=index, memory_key='snippets', input_key="snippets")
    
    # memory is a combination of the last 3 messages and the last 3 snippets
    memory = CombinedMemory(memories=[conv_memory, snippets_memory])

    conversation = construct_conversation(prompt_header, llm, memory)

    return conversation
