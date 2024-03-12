from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )

instruction = """You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible. Infer the income of the user from the below conversation between the user and the assistant. """
system = """You know, I was thinking today about my last visit to the barber. Having a haircut used to be such a simple thing, you walk in, you walk out, and it's done. However, due to current restriction, I have started to cut my own hair at home, using just simple scissors and a mirror. Saved me a decent chunk as I don't have to tip the barber now! Haha! Tell me, did you ever have to manage your own haircut or grooming? If so, how was your experience with that?""".encode('utf-8').decode('utf-8', 'replace').replace('\ufffd', '')
prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
for i in llm(prompt, stream=True):
    print(i, end="", flush=True)
print()
