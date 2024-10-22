import pandas as pd
import random
import json
from anthropic import Anthropic
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict
import asyncio
from anthropic import AsyncAnthropic
import sys

# Replace with your actual Anthropic API key
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")

def load_data(file_path):
    return pd.read_csv(file_path)

def create_prompt(input_text, output_1, output_2):
    prompt = f"""Your task is to compare outputs from two Large Language Models (LLMs) and determine which one is superior, or if they are equally good. You will be provided with an input prompt and two corresponding outputs. Analyze these outputs thoroughly using the following criteria and structure. Fully restate each criteria before proceeding. i.e. "1. Relevance and Accuracy: How well ...":

1. Relevance and Accuracy:
How well does each output address the specific requirements in the input prompt?
Is the information provided correct and consistent with the given details?

2. Comprehensiveness and Depth:
Does each output cover all aspects mentioned in the input prompt?
How thorough and insightful is the evaluation provided?

3. Structure and Clarity:
How well-organized and easy to understand is each output?
Does it follow any specific format requirements mentioned in the input prompt?

For each criterion, provide a brief comparison of both outputs. Then, summarize the strengths and weaknesses of each output based on your analysis.
Finally, make your decision: choose output_1, output_2, or declare a tie. Explain your reasoning for this choice.
Present your final decision in this standardized format:
<decision> {{ "explanation": "[explanation of your decision]", "decision": "[output_1/output_2/tie]"}} </decision>
It is crucial to remain objective throughout your analysis. Base your decision solely on the quality and effectiveness of the outputs in relation to the given input prompt. Use specific examples from each output to support your evaluation.
Here are the elements for your analysis:
<input> {input_text} </input> 
<output_1> {output_1} </output_1> 
<output_2> {output_2} </output_2>"""
    return prompt


async def get_claude_evaluation(client, prompt):
    response = await client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content


def parse_decision(response):
    response = response[0].text
    decision_start = response.find('<decision>') + len('<decision>')
    decision_end = response.find('</decision>')
    if decision_start != -1 and decision_end != -1:
        decision_json = response[decision_start:decision_end].strip()
        try:
            decision = json.loads(decision_json)
            return decision['decision'], decision['explanation']
        except json.JSONDecodeError:
            return "error", "Failed to parse decision JSON"
    return "error", "Decision not found in response"


async def process_batch(client, batch):
    tasks = []
    for row in batch:
        input_text = row['input']
        generated_text = row['generated_text']
        answer = row['answer']
        
        if random.choice([True, False]):
            output_1, output_2 = generated_text, answer
            output_1_label, output_2_label = "generated_text", "answer"
        else:
            output_1, output_2 = answer, generated_text
            output_1_label, output_2_label = "answer", "generated_text"
        
        prompt = create_prompt(input_text, output_1, output_2)
        task = asyncio.ensure_future(get_claude_evaluation(client, prompt))
        tasks.append((task, row['model'], output_1_label, output_2_label))
    
    responses = await asyncio.gather(*[t[0] for t in tasks])
    
    results = []
    for response, (_, model, output_1_label, output_2_label) in zip(responses, tasks):
        decision, explanation = parse_decision(response)
        
        if decision == "output_1":
            winner = output_1_label
        elif decision == "output_2":
            winner = output_2_label
        else:
            winner = "tie"
        
        results.append({
            'model': model,
            'winner': winner,
            'explanation': explanation
        })
    
    return results


async def evaluate_models(df, batch_size=4):
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    results = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size].to_dict('records')
        batch_results = await process_batch(client, batch)
        results.extend(batch_results)
    
    return pd.DataFrame(results)


def analyze_results(results_df):
    # Calculate win rates for each model
    win_rates = results_df[results_df['winner'] == 'generated_text'].groupby('model').size() / results_df.groupby('model').size()
    win_rates = win_rates.reset_index()
    win_rates.columns = ['model', 'win_rate']
    
    # Calculate tie rates for each model
    tie_rates = results_df[results_df['winner'] == 'tie'].groupby('model').size() / results_df.groupby('model').size()
    tie_rates = tie_rates.reset_index()
    tie_rates.columns = ['model', 'tie_rate']
    
    # Merge win rates and tie rates
    analysis = pd.merge(win_rates, tie_rates, on='model', how='outer').fillna(0)
    analysis['loss_rate'] = 1 - analysis['win_rate'] - analysis['tie_rate']
    
    return analysis

def plot_results(analysis):
    plt.figure(figsize=(12, 6))
    x = range(len(analysis))
    width = 0.25
    
    plt.bar([i - width for i in x], analysis['win_rate'], width, label='Win Rate', color='green')
    plt.bar(x, analysis['tie_rate'], width, label='Tie Rate', color='yellow')
    plt.bar([i + width for i in x], analysis['loss_rate'], width, label='Loss Rate', color='red')
    
    plt.xlabel('Model')
    plt.ylabel('Rate')
    plt.title('Model Performance Comparison')
    plt.xticks(x, analysis['model'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison_results.png')
    plt.close()

async def main():
    
    file_path = sys.argv[1]
    
    # Load data
    if not file_path:
        raise ValueError("Please provide a file path to the model outputs")
    
    df = load_data(file_path)
    
    # Evaluate models
    results_df = await evaluate_models(df)
    
    # Analyze results
    analysis = analyze_results(results_df)
    
    # Plot results
    plot_results(analysis)
    
    # Print analysis
    print(analysis)
    
    # Save results into 'evaluation_results.csv'. If file already exists, append to it.
    evaluation_results_path = 'evaluation_results.csv'
    if os.path.exists(evaluation_results_path):
        results_df.to_csv(evaluation_results_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(evaluation_results_path, index=False)
    
    # Save analysis into 'analysis_results.csv'. If file already exists, append to it.
    analysis_results_path = 'analysis_results.csv'
    if os.path.exists(analysis_results_path):
        analysis.to_csv(analysis_results_path, mode='a', header=False, index=False)
    else:
        analysis.to_csv(analysis_results_path, index=False)
    


if __name__ == "__main__":
    asyncio.run(main())
