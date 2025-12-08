import re

def extract_answer(response):
    # extract everything between ```python and ```
    return response.split("```python")[1].split("```")[0]

def extract_answer2(response):
    answer1 = extract_answer(response)
    answer2 = re.sub(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|#.*?$)', '', answer1, flags=re.DOTALL | re.MULTILINE)
    answer2 = "\n".join([line for line in answer2.split("\n") if line.strip()]) # remove any empty lines
    return answer2
