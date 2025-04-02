import json_repair
from openai import OpenAI


client = OpenAI()


def get_match_description(match_info):
    source_domain = match_info["source_domain"]
    source_values = []
    if "value_name" in source_domain.columns:
        source_values = source_domain["value_name"].tolist()

    target_domain = match_info["target_domain"]
    target_values = []
    if "value_name" in target_domain.columns:
        target_values = target_domain["value_name"].tolist()

    if match_info["method_description"] is None:
        method_description = ""
    else:
        method_description = match_info["method_description"]

    if match_info["use_method_info"]:
        description = (
            f"The method that was used is {match_info['method_name']}. {method_description}. "
            f"Source column: {match_info['source_column']}, unique values: {source_values}.\n"
            f"Target column: {match_info['target_column']}, unique values: {target_values}.\n"
            f"Similarity score: {match_info['similarity']}"
        )
    else:
        description = (
            f"Source column: {match_info['source_column']}, unique values: {source_values}.\n"
            f"Target column: {match_info['target_column']}, unique values: {target_values}.\n"
            f"Similarity score: {match_info['similarity']}"
        )

    return description


def evaluate_match(match_info):
    match_description = get_match_description(match_info)

    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "Your task is to evaluate the matching between two columns from a source and a target dataset. "
                "This matching is performed by an automatic method. "
                "I will provide the column names, a similarity score, and their unique values (if available). "
                "I will also provide the name of the method and how it works to find matches. "
                "Your response should be a Python dictionary with exactly two fields: "
                "'response': Use only one of the values 'accept', 'need review' or 'reject'. "
                "'explanation': Provide a concise justification for your decision including the logic of the method. "
                "Return only the Python dictionary as output, with no additional text. Ensure your response excludes quotations",
            },
            {
                "role": "user",
                "content": match_description,
            },
        ],
    )

    response_message = completion.choices[0].message.content
    try:
        response_dict = json_repair.loads(response_message)
        decision_class = response_dict["response"]
        decision_explanation = response_dict["explanation"]
        return decision_class, decision_explanation
    except:
        print(f"Errors parsing response '{response_message}'")
        return {"response": "unknown", "explanation": "unknown"}
