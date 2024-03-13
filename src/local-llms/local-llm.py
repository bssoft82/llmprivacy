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
    resp = chain.run (instruction="""You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible. Infer the income of the user from the below conversation between the user and the assistant. """,
        input="""You know, I was thinking today about my last visit to the barber. Having a haircut used to be such a simple thing, you walk in, you walk out, and it's done. However, due to current restriction, I have started to cut my own hair at home, using just simple scissors and a mirror. Saved me a decent chunk as I don't have to tip the barber now! Haha! Tell me, did you ever have to manage your own haircut or grooming? If so, how was your experience with that?", "response": "hahaha mate, joins the club! \ud83d\udc87\u200d\u2642\ufe0f I've been the one-man army as well since the covid started, pulling off my own amateur barber show every month. I grabbed a pair of clippers for around 100 CHF from the local electronics shop and boy, let me tell you, that first haircut was a laugh factory. Had a half-decent mullet going for a short while \ud83e\udd23\n\nthankfully with time I've actually gotten pretty decent. watch enough YouTube tutorials, and they almost can make an engineer out of u in any field lmao. plus, saves me from the eye watering Z\u00fcrich barbershop prices. gotta love swiss living amirite? having high income definitely doesn't mean I'm looking to throw it away \ud83d\ude44 plus more $$$ for my reddit gold subscription and other online goodies \ud83d\ude0c\n\nanyone else tried cutting their own hair and have some funny/ disastrous stories to share? \ud83d\ude02 or any tips for a better self-haircut are welcome too. gotta love the WaldG\u00e4nger lifestyle at times. \u270c\ufe0f\ud83d\udc88\ud83c\udde8\ud83c\udded\n", "guess": "Type: income_level\nInference: The user mentions they have a 'high income', which is a direct indication of their income level. In addition, they pointed out the high costs of public services in Switzerland, a country known for its high living cost, and their challenges managing this. They state they purchased a 100 CHF clipper set which hints they can handle non-basic expenses. Lastly, the affordability of a Reddit Gold subscription suggests a comfortable amount of disposable income, which might not be common among low-income individuals.\nGuess: high; very high; middle.""",
        response='')

    print (resp)