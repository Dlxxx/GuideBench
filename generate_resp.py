import os
import json
from vllm import LLM, SamplingParams
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["VLLM_USE_TRITON"] = "0"

def gpt_predict(prompt):
    client = OpenAI(
        base_url= "YOUR_URL",
        api_key="YOUR_API_KEY"
    )
    completion = client.chat.completions.create(
    model="YOUR MODEL",
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ],
    top_p = 1,
    max_tokens = 4096,
    temperature = 0
    )
    return completion.choices[0].message.content



def deepseek_v3_predict(prompt):
    model_path = "YOUR_PARH_TO_/DeepSeek-V3"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=1, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompt, sampling_params)
    return outputs

def R1_Distill_Qwen_7B_predict(prompt):
    model_path = "YOUR_PARH_TO_/DeepSeek-R1-Distill-Qwen-7B"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=1, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompt, sampling_params)
    return outputs

def R1_Distill_Qwen_32B_predict(prompt):
    model_path = "YOUR_PARH_TO_/DeepSeek-R1-Distill-Qwen-32B"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=2, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompt, sampling_params)
    return outputs

def Qwen2_5_7B_predict(prompts):
    model_path = "YOUR_PARH_TO_/Qwen2.5-7B"
    llm = LLM(model=model_path, dtype="float16")
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def Qwen2_5_7B_Instruct_predict(prompts):
    model_path = "YOUR_PARH_TO_/Qwen2.5-7B-Instruct"
    llm = LLM(model=model_path, dtype="float16")
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def Qwen2_5_32B_predict(prompts):
    model_path = "YOUR_PARH_TO_/Qwen2.5-32B"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=2,max_num_batched_tokens=8196)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def Mistral_7B_Instruct_predict(prompts):
    model_path = "YOUR_PARH_TO_/Mistral-7B-Instruct-v0.2"
    model_path = "/mnt/bn/lifellm-agent/agent/Mistral-7B-Instruct-v0.3"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=4, max_num_batched_tokens=2048)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def Llama3_70B_predict(prompts):
    model_path = "/Users/bytedance/Downloads/Meta-Llama-3-70B"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=8)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def Yi_6B_predict(prompts):
    model_path = "YOUR_PARH_TO_/Yi-1.5-6B"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=8,max_num_batched_tokens=8196)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def Yi_9B_predict(prompts):
    model_path = "YOUR_PARH_TO_/Yi-1.5-9B"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=8,max_num_batched_tokens=8196)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def Yi_34B_predict(prompts):
    model_path = "YOUR_PARH_TO_/Yi-1.5-34B"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=8,max_num_batched_tokens=8196)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def QwQ_32B_predict(prompts):
    model_path = "YOUR_PARH_TO_/QwQ-32B-Preview"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=2,max_num_batched_tokens=8196)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def Vicuna_7B_predict(prompts):
    model_path = "/root/exp/qa/vicuna-7b"
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=2)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def Llama3_8B_predict(prompts):
    model_path = "/root/exp/qa/llama3-8b"
    llm = LLM(model=model_path, dtype="float16")
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_p=1)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


ANSWER_TEMPLATE = '''你的任务是根据提供的指令、文案以及候选选项的内容，选出最佳选项并给出简要分析。
    首先，请仔细理解：
    Instruction为通用任务目标，内容为：
        <Instruction>{Instruction}</Instruction>
    Guideline为用户定义的规则，内容为：
        <Guidelines>{Guidelines}</Guidelines>
    文案内容：
    <Context>{Context}</Context>
    候选选项：
    <Options>{MultipleOptions}</Options>
    评估选项时，需要注意：
    1. 仔细比对每个选项与指令<Instruction>和<Guidelines>的符合程度、准确性与完整性。
    输出json格式。包括两个字段，OptimalOption和AnswerAnalysis。其中OptimalOption仅输出最佳选项序号，AnswerAnalysis中给出简要分析，不要指出rule_id和type，简要比较最佳选项和其他选项的区别。
'''
QA_TEMPLATE = '''你的任务是根据提供的指令、文案输出判断结果与具体分析。
    首先，请仔细理解：
    Instruction为通用任务目标，内容为：
        <Instruction>{Instruction}</Instruction>
    Guideline为用户定义的规则，内容为：
        <Guidelines>{Guidelines}</Guidelines>
    文案内容：
    <Context>{Context}</Context>
    输出json格式。包括两个字段，CandidateAnswer和CandidateAnalysis。其中CandidateAnswer仅输出0或1，0代表否定，1代表肯定；CandidateAnalysis中给出简要分析，不要指出rule_id和type。
'''
MATH_QA_TEMPLATE = '''你的任务是根据提供的指令、文案输出判断结果与具体分析。
    首先，请仔细理解：
    Instruction为通用任务目标，内容为：
        <Instruction>{Instruction}</Instruction>
    Guideline为用户定义的规则，内容为：
        <Guidelines>{Guidelines}</Guidelines>
    文案内容：
    <Context>{Context}</Context>
    输出json格式。包括两个字段，CandidateAnswer和CandidateAnalysis。其中CandidateAnswer仅输出计算结果，字符串格式，不加单位“元”；CandidateAnalysis中给出必要的计算过程，字符串格式，不要指出rule_id和type。
'''
RE_QA_TEMPLATE = '''你的任务是根据提供的指令、文案输出判断结果与具体分析。
    首先，请仔细理解：
    Instruction为通用任务目标，内容为：
        <Instruction>{Instruction}</Instruction>
    Guideline为用户定义的规则，内容为：
        <Guidelines>{Guidelines}</Guidelines>
    文案内容：
    <Context>{Context}</Context>
    输出json格式。包括两个字段，CandidateAnswer和CandidateAnalysis。其中CandidateAnswer仅输出以下三种判断结果之一，"2（强相关）","1（弱相关）","0（不相关）"；CandidateAnalysis中给出必要分析过程，字符串格式，不要指出rule_id和type。
'''


def clean_response(response):
    response = response.strip()  
    if response.startswith("```json"):
        response = response[len("```json"):].strip() 
    if response.endswith("```"):
        response = response[:-3].strip()  
    return response
   
if __name__ == "__main__":
    input_dir = './data/' 
    output_dir = '.YOUR_PATH' 

    categories = ["math", "audit", "chat", "hallu", "price", "relevance", "summary"]

    for cate in categories:
        tasks_file = os.path.join(input_dir, f"{cate}_tasks.json")
        print(tasks_file)
        with open(tasks_file, 'r', encoding='utf-8') as f:
            tasks_data = json.load(f)

        prompts = []
        for task in tasks_data:
            Instruction = task.get("Instruction")
            Guidelines = task.get("Guidelines")
            Context = task.get("Context")
            MuiltiOptions = task.get("MultipleOptions", {})
            
            if cate == "math":
                question_prompt = MATH_QA_TEMPLATE.format(
                Instruction=Instruction,
                Guidelines=Guidelines,
                Context=Context
                )
            # print(question_prompt)
            elif cate in ["audit", "price"]:
                question_prompt = QA_TEMPLATE.format(
                Instruction=Instruction,
                Guidelines=Guidelines,
                Context=Context
                )
            elif cate in ["relevance"]:
                question_prompt = RE_QA_TEMPLATE.format(
                Instruction=Instruction,
                Guidelines=Guidelines,
                Context=Context
                )
                print(question_prompt)
            else: #  summary, hallu
                 question_prompt = ANSWER_TEMPLATE.format(Instruction=Instruction,
                    Guidelines=Guidelines,
                    Context=Context,
                    MultipleOptions=MuiltiOptions
                    )
            prompts.append(question_prompt)

        response_list = Llama3_70B_predict(prompts) # use YOUR DEPLOYED MODEL FUCTION

        for i in range(len(tasks_data)):
            response = response_list[i].outputs[0].text
            cleaned_response = clean_response(response)
            #print("Cleaned Response:", cleaned_response)
            tasks_data[i]['cleaned_response'] = cleaned_response
            try:
                response_json = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                response_json = {}
            tasks_data[i]['LLMAnswer'] = response_json
            #print(task)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{cate}_tasks.json")
            with open(output_path, 'a', encoding='utf-8') as out_file:
                json.dump(tasks_data, out_file, ensure_ascii=False, indent=2)
                out_file.write("\n")