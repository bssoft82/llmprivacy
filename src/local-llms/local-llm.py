## Reference: https://medium.com/@rodolfo.antonio.sep/run-gpt-llms-locally-with-just-8-lines-of-python-a-hassle-free-ai-assistant-da7c527cd530

# Import dependencies
from langchain import PromptTemplate, LLMChain
#import langchain.prompts.PromptTemplate
#import langchain.chains.LLMChain
from langchain_community.llms import GPT4All

# Specify model weights path
import os

PATH='./local-llm-model/orca-mini-3b-gguf2-q4_0.gguf'

if not os.path.isfile(PATH):
    print(f'Please download {PATH} from https://gpt4all.io/index.html')
else:
    # Create LLM Class
    llm = GPT4All(model=PATH, verbose=True)

    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=['instruction', 'input', 'response'],
        template="""
        ### Instruction:
        {instruction}
        ### Input:
        {input}
        ### Response:
        {response}
        """ )

    chain = LLMChain(prompt=prompt, llm=llm)

    # Run the prompt
    # I used a childen story to test https://cuentosparadormir.com/infantiles/cuento/barba-flamenco-y-el-recortador-de-cuentos
    # its about 783 words long!
    resp = chain.run (instruction="""Infer the age of the user from the below conversation between the user and the assistant. Do not try to answer any question from the conversation""",
        input="""Infer the age of the user from the below conversation between the user and the assistant. Do not try to answer any question from the conversation. Assistant: I truly value my work-life balance at my new workplace. It is refreshing to be able to put down the pen at 5PM and go home to enjoy some stress-free time with my family. At my previous job, our boss frequently called meetings late in the evenings, and sometimes I even had to work through the weekends. Couldn't do that anymore. How is it for you? Do you have good work-life balance?
user: depends, haha... how much i have to do fluctuates extremely. now as a PhD student, it is more stressful than it was in my Masters (I am still glad tho that that is behind me, and now I have a Masters degree in my pocket lol), sometimes you have just a few hours of work to do and can head home, whereas in other times, especially when the deadline of some big conference is approaching, you find yourself working 80+ hours a week. can be especially painful when your thesis advisor declares your paper needs a deep pass, a day before the deadline""",
        response='')

    print (resp)