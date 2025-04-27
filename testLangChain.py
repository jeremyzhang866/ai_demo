from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import os
os.environ["OPENAI_API_KEY"] = '...'

llm = OpenAI(temperature=0)
mem = ConversationBufferMemory()

# Here it is by default set to "AI"
conversation = ConversationChain(llm=llm, verbose=True, memory=mem)

conversation.predict(input="Hi there!")